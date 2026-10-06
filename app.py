import gspread
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from google.oauth2.service_account import Credentials
from scipy.stats import poisson
from urllib.parse import quote

st.set_page_config(page_title="Analizador de partidos", page_icon="⚽")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets",
          "https://www.googleapis.com/auth/drive"]
COLS = ["Fecha", "Liga", "Local", "Visitante", "Mercado", "Seleccion",
        "Cuota", "Monto", "Prob_modelo", "Resultado", "Ganancia"]
CERRADAS = ["Ganada", "Perdida", "Nula"]

# ---------- Modelo y datos ----------
@st.cache_resource
def cargar():
    p = joblib.load("modelo_apuestas.joblib")
    h = pd.read_csv("historial.csv", parse_dates=["Date"])
    return (p["modelo"], p["columnas"], p["elo"], h,
            p["goles_local"], p["goles_visitante"])

modelo, columnas, elo, h, gol_l, gol_v = cargar()

def historial_equipo(liga, equipo):
    d = h[(h.Liga == liga) & ((h.HomeTeam == equipo) | (h.AwayTeam == equipo))].sort_values("Date")
    loc = d.HomeTeam == equipo
    o = pd.DataFrame({"Date": d.Date, "Sede": np.where(loc, "L", "V"),
                      "GF": np.where(loc, d.FTHG, d.FTAG), "GC": np.where(loc, d.FTAG, d.FTHG)})
    o["Pts"] = np.where(o.GF > o.GC, 3, np.where(o.GF == o.GC, 1, 0))
    return o

def calcular(liga, local, visitante):
    fecha = pd.Timestamp.today().normalize()
    f, forma = {}, {}
    for p, eq, sede in [("L_", local, "L"), ("V_", visitante, "V")]:
        d = historial_equipo(liga, eq)
        u5, u10 = d.tail(5), d.tail(10)
        f[p + "GF_5"], f[p + "GC_5"], f[p + "Pts_5"] = u5.GF.mean(), u5.GC.mean(), u5.Pts.mean()
        f[p + "GD_10"] = (u10.GF - u10.GC).mean()
        f[p + "PtsSede_5"] = d[d.Sede == sede].tail(5).Pts.mean()
        f[p + "Descanso"] = min((fecha - d.Date.iloc[-1]).days, 14)
        f[p + "Elo"] = elo.get((liga, eq), 1500)
        forma[eq] = (u5.Pts.sum(), u5.GF.sum(), u5.GC.sum())
    f["Dif_Elo"] = f["L_Elo"] - f["V_Elo"]
    f["Liga_" + liga] = 1.0
    x = pd.DataFrame([f]).reindex(columns=columnas, fill_value=0.0)
    pa, pe, ph = modelo.predict_proba(x)[0]   # orden: A, D, H
    glocal = float(gol_l.predict(x)[0])
    gvisit = float(gol_v.predict(x)[0])
    return ph, pe, pa, f, forma, glocal, gvisit

# ---------- Google Sheets ----------
@st.cache_resource
def libro():
    creds = Credentials.from_service_account_info(
        dict(st.secrets["gcp_service_account"]), scopes=SCOPES)
    return gspread.authorize(creds).open_by_key(st.secrets["sheet_id"])

@st.cache_data(ttl=60)
def leer_apuestas():
    filas = libro().worksheet("apuestas").get_all_records()
    df = pd.DataFrame(filas, columns=COLS)
    for c in ["Cuota", "Monto", "Prob_modelo", "Ganancia"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

@st.cache_data(ttl=300)
def leer_bank_inicial():
    v = libro().worksheet("config").acell("B1").value
    return float(str(v).replace(",", ""))

def guardar_apuesta(fila):
    libro().worksheet("apuestas").append_row(fila, value_input_option="RAW")
    leer_apuestas.clear()

def cerrar_apuesta(idx, resultado, ganancia):
    ws = libro().worksheet("apuestas")
    ws.update_cell(idx + 2, 10, resultado)     # columna J: Resultado
    ws.update_cell(idx + 2, 11, float(ganancia))  # columna K: Ganancia
    leer_apuestas.clear()

# ---------- Cierre automático ----------
def leer_fecha(x):
    f = pd.to_datetime(str(x), format="%Y-%m-%d", errors="coerce")
    if pd.isna(f):
        f = pd.to_datetime(str(x), dayfirst=True, errors="coerce")
    return f

def resultado_apuesta(fila, partido):
    gl, gv = partido.FTHG, partido.FTAG
    total = gl + gv
    sel = str(fila.Seleccion)
    if sel == f"Gana {fila.Local}":
        gano = gl > gv
    elif sel == f"Gana {fila.Visitante}":
        gano = gv > gl
    elif sel == "Empate":
        gano = gl == gv
    elif sel == "Local o empate":
        gano = gl >= gv
    elif sel == "Local o visitante":
        gano = gl != gv
    elif sel == "Empate o visitante":
        gano = gv >= gl
    elif sel.startswith("Over "):
        gano = total > float(sel.split()[1])
    elif sel.startswith("Under "):
        gano = total < float(sel.split()[1])
    else:
        return None
    return "Ganada" if gano else "Perdida"

def cerrar_automaticas(df):
    cerradas, notas = 0, []
    for i, fila in df[df.Resultado == "Pendiente"].iterrows():
        nombre = f"{fila.Local} vs {fila.Visitante} ({fila.Seleccion})"
        fecha = leer_fecha(fila.Fecha)
        if pd.isna(fecha):
            notas.append(f"{nombre}: no pude leer la fecha '{fila.Fecha}'")
            continue
        desde, hasta = fecha - pd.Timedelta(days=1), fecha + pd.Timedelta(days=30)
        cand = h[(h.Liga == fila.Liga) & (h.HomeTeam == fila.Local)
                 & (h.AwayTeam == fila.Visitante)
                 & (h.Date >= desde) & (h.Date <= hasta)].sort_values("Date")
        if cand.empty:
            notas.append(f"{nombre}: no hay resultado en los datos entre "
                         f"{desde.date()} y {hasta.date()} (liga '{fila.Liga}')")
            continue
        res = resultado_apuesta(fila, cand.iloc[0])
        if res is None:
            notas.append(f"{nombre}: no reconozco la selección '{fila.Seleccion}'")
            continue
        gan = fila.Monto * (fila.Cuota - 1) if res == "Ganada" else -fila.Monto
        cerrar_apuesta(i, res, gan)
        cerradas += 1
    st.session_state["notas"] = notas
    return cerradas

try:
    libro()
    sheets_ok = True
except Exception as e:
    sheets_ok = False
    st.warning(f"No pude conectar con Google Sheets, así que el registro no está disponible. Detalle: {e}")

df, inicial, bank = None, None, None
if sheets_ok:
    df = leer_apuestas()
    if not st.session_state.get("auto_revisado"):
        st.session_state["auto_revisado"] = True
        try:
            n = cerrar_automaticas(df)
            if n:
                df = leer_apuestas()
                st.toast(f"Se cerraron {n} apuestas automáticamente")
        except Exception as e:
            st.warning(f"No pude revisar los resultados automáticamente: {e}")
    inicial = leer_bank_inicial()
    bank = inicial + df[df.Resultado.isin(CERRADAS)].Ganancia.sum()

# ---------- Noticias ----------
ALIAS = {
    "Guadalajara Chivas": "Chivas", "UNAM Pumas": "Pumas", "Atl. San Luis": "Atlético San Luis",
    "Club Leon": "León", "Club America": "América", "Queretaro": "Querétaro",
    "Juarez": "FC Juárez", "Man City": "Manchester City", "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest", "Ath Madrid": "Atlético de Madrid",
    "Ath Bilbao": "Athletic Club", "Sociedad": "Real Sociedad", "Paris SG": "PSG",
    "M'gladbach": "Borussia Mönchengladbach", "Ein Frankfurt": "Eintracht Frankfurt",
    "Inter": "Inter de Milán", "Milan": "AC Milan", "Betis": "Real Betis",
    "St. Louis City": "St. Louis City SC",
}

@st.cache_data(ttl=1800)
def noticias(equipo, liga):
    nombre = ALIAS.get(equipo, equipo)
    q = f"{nombre} {liga} (lesión OR lesionado OR baja OR duda OR suspendido) when:7d"
    url = f"https://news.google.com/rss/search?q={quote(q)}&hl=es-419&gl=MX&ceid=MX:es-419"
    r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    items = ET.fromstring(r.content).findall(".//item")[:6]
    return [(i.findtext("title"), i.findtext("link")) for i in items]

# ---------- Panel lateral: convertidor de momios ----------
with st.sidebar:
    st.subheader("Convertidor de momios")
    am = st.number_input("Momio americano (ej. -150 o +200)", value=0, step=5)
    if am >= 100:
        st.write(f"Decimal: **{1 + am / 100:.2f}**")
    elif am <= -100:
        st.write(f"Decimal: **{1 + 100 / abs(am):.2f}**")
    else:
        st.caption("Escribe un momio de +100 en adelante o de -100 hacia abajo.")

# ---------- Interfaz ----------
st.title("⚽ Analizador de partidos")
st.caption("Herramienta de análisis y práctica con dinero virtual. No es consejo de apuestas.")
tab1, tab2, tab3 = st.tabs(["🔎 Analizar", "📝 Mis apuestas", "📊 Panel"])

# ===== Pestaña 1: Analizar =====
with tab1:
    liga = st.selectbox("Liga", sorted(h.Liga.unique()))
    recientes = h[(h.Liga == liga) & (h.Date >= h.Date.max() - pd.Timedelta(days=365))]
    equipos = sorted(set(recientes.HomeTeam) | set(recientes.AwayTeam))
    local = st.selectbox("Local", equipos)
    visitante = st.selectbox("Visitante", [e for e in equipos if e != local])

    if st.button("Analizar", type="primary"):
        ph, pe, pa, f, forma, glocal, gvisit = calcular(liga, local, visitante)
        st.session_state["an"] = dict(liga=liga, local=local, visitante=visitante,
                                      ph=ph, pe=pe, pa=pa, f=f, forma=forma,
                                      glocal=glocal, gvisit=gvisit)

    an = st.session_state.get("an")
    if an and "glocal" not in an:
        an = None
    if an:
        ph, pe, pa = an["ph"], an["pe"], an["pa"]
        lam = an["glocal"] + an["gvisit"]
        p_over = lambda linea: 1 - poisson.cdf(int(linea), lam)

        st.subheader(f"{an['local']} vs {an['visitante']}")
        a, b, c = st.columns(3)
        a.metric("Gana local", f"{ph:.1%}")
        b.metric("Empate", f"{pe:.1%}")
        c.metric("Gana visitante", f"{pa:.1%}")
        st.write(f"**Doble oportunidad:** Local o empate {ph+pe:.1%} · "
                 f"Local o visitante {ph+pa:.1%} · Empate o visitante {pe+pa:.1%}")
        st.write(f"**Goles esperados:** {an['local']} {an['glocal']:.2f} · "
                 f"{an['visitante']} {an['gvisit']:.2f} · total {lam:.2f}")
        st.write(f"**Over:** 1.5 → {p_over(1.5):.1%} · 2.5 → {p_over(2.5):.1%} · 3.5 → {p_over(3.5):.1%}")
        st.write(f"**Elo:** {an['local']} {an['f']['L_Elo']:.0f} · "
                 f"{an['visitante']} {an['f']['V_Elo']:.0f}")
        for eq, (pts, gf, gc) in an["forma"].items():
            st.write(f"**Últimos 5 de {eq}:** {pts} puntos, {gf} goles a favor, {gc} en contra")

        if st.button("📰 Noticias recientes (lesiones y bajas)"):
            for eq in (an["local"], an["visitante"]):
                st.markdown(f"**{eq}**")
                try:
                    items = noticias(eq, an["liga"])
                    if not items:
                        st.caption("Sin noticias recientes de lesiones o bajas.")
                    for t, l in items:
                        st.markdown(f"- [{t}]({l})")
                except Exception as e:
                    st.caption(f"No pude cargar las noticias: {e}")

        st.markdown("#### Cuotas del mercado (deja en 0 las que no uses)")
        c1, c2, c3 = st.columns(3)
        cl = c1.number_input("Local", min_value=0.0, value=0.0, step=0.05, key="cl")
        ce = c2.number_input("Empate", min_value=0.0, value=0.0, step=0.05, key="ce")
        cv = c3.number_input("Visitante", min_value=0.0, value=0.0, step=0.05, key="cv")
        d1, d2, d3 = st.columns(3)
        c1x = d1.number_input("Local o empate", min_value=0.0, value=0.0, step=0.05, key="c1x")
        c12 = d2.number_input("Local o visitante", min_value=0.0, value=0.0, step=0.05, key="c12")
        cx2 = d3.number_input("Empate o visitante", min_value=0.0, value=0.0, step=0.05, key="cx2")
        e1, e2, e3 = st.columns(3)
        ov15 = e1.number_input("Over 1.5", min_value=0.0, value=0.0, step=0.05, key="ov15")
        ov25 = e2.number_input("Over 2.5", min_value=0.0, value=0.0, step=0.05, key="ov25")
        ov35 = e3.number_input("Over 3.5", min_value=0.0, value=0.0, step=0.05, key="ov35")
        g1, g2, g3 = st.columns(3)
        un15 = g1.number_input("Under 1.5", min_value=0.0, value=0.0, step=0.05, key="un15")
        un25 = g2.number_input("Under 2.5", min_value=0.0, value=0.0, step=0.05, key="un25")
        un35 = g3.number_input("Under 3.5", min_value=0.0, value=0.0, step=0.05, key="un35")

        sel = {
            f"Gana {an['local']}": (ph, cl, "1X2"),
            "Empate": (pe, ce, "1X2"),
            f"Gana {an['visitante']}": (pa, cv, "1X2"),
            "Local o empate": (ph + pe, c1x, "Doble oportunidad"),
            "Local o visitante": (ph + pa, c12, "Doble oportunidad"),
            "Empate o visitante": (pe + pa, cx2, "Doble oportunidad"),
            "Over 1.5": (p_over(1.5), ov15, "Over/Under"),
            "Over 2.5": (p_over(2.5), ov25, "Over/Under"),
            "Over 3.5": (p_over(3.5), ov35, "Over/Under"),
            "Under 1.5": (1 - p_over(1.5), un15, "Over/Under"),
            "Under 2.5": (1 - p_over(2.5), un25, "Over/Under"),
            "Under 3.5": (1 - p_over(3.5), un35, "Over/Under"),
        }
        filas = []
        for nombre, (prob, cuota, mercado) in sel.items():
            if cuota > 0:
                ev = prob * cuota
                filas.append({"Selección": nombre, "Mercado": mercado,
                              "Prob. modelo": f"{prob:.1%}", "Cuota": f"{cuota:.2f}",
                              "Valor esperado": f"{ev:.2f}",
                              "Sugerencia": "Apostar" if ev >= 1.05 else "Pasar"})
        if filas:
            st.markdown("#### Sugerencias")
            st.dataframe(pd.DataFrame(filas), hide_index=True)
            st.caption("Regla: se sugiere apostar si el valor esperado es al menos 1.05. "
                       "En pruebas con datos pasados, esta regla perdió bastante en el resultado del "
                       "partido (ROI de -14%) y quedó cerca del empate en over/under 2.5 (ROI de -0.2%, "
                       "con un margen de error de unos 3 puntos). Úsala como práctica, no como señal.")

            if sheets_ok:
                st.markdown("#### Registrar apuesta simulada")
                elegida = st.selectbox("Selección", [fila["Selección"] for fila in filas])
                sugerido = round(bank * 0.01, 2)
                monto = st.number_input("Monto (1% del bank por defecto)", min_value=0.0,
                                        value=sugerido, step=1.0, key=f"monto_{bank:.2f}")
                st.caption(f"Bank actual: {bank:,.2f} · Tope por apuesta (2%): {bank*0.02:,.2f}")
                if st.button("Registrar apuesta"):
                    prob, cuota, mercado = sel[elegida]
                    if monto <= 0:
                        st.error("El monto debe ser mayor que 0.")
                    elif monto > bank * 0.02:
                        st.error("El monto supera el 2% del bank. Baja el monto.")
                    else:
                        guardar_apuesta([pd.Timestamp.today().strftime("%Y-%m-%d"), an["liga"],
                                         an["local"], an["visitante"], mercado, elegida,
                                         float(cuota), float(monto), float(prob), "Pendiente", 0])
                        st.success("Apuesta registrada. Se cerrará sola cuando el resultado del partido esté en los datos.")

# ===== Pestaña 2: Mis apuestas =====
with tab2:
    if not sheets_ok:
        st.info("Conecta Google Sheets para usar el registro.")
    else:
        aviso = st.session_state.pop("aviso", None)
        if aviso:
            st.success(aviso)
        pend = df[df.Resultado == "Pendiente"]
        st.subheader("Apuestas pendientes")
        if st.button("🔄 Revisar resultados ahora"):
            n = cerrar_automaticas(df)
            st.session_state["aviso"] = (f"Se cerraron {n} apuestas automáticamente." if n
                                         else "No hay resultados nuevos para tus apuestas pendientes.")
            st.rerun()
        for nota in st.session_state.get("notas", []):
            st.caption("• " + nota)
        if pend.empty:
            st.info("No tienes apuestas pendientes.")
        else:
            opciones = {i: f"{r.Local} vs {r.Visitante} · {r.Seleccion} · cuota {r.Cuota:.2f} · monto {r.Monto:.2f}"
                        for i, r in pend.iterrows()}
            st.caption("Se cierran solas cuando el resultado aparece en los datos. "
                       "Si un partido se suspendió, ciérralo aquí a mano como Nula.")
            i = st.selectbox("Cerrar a mano", list(opciones), format_func=lambda k: opciones[k])
            res = st.radio("Resultado", CERRADAS, horizontal=True)
            if st.button("Cerrar apuesta"):
                fila = df.loc[i]
                if res == "Ganada":
                    gan = fila.Monto * (fila.Cuota - 1)
                elif res == "Perdida":
                    gan = -fila.Monto
                else:
                    gan = 0.0
                cerrar_apuesta(i, res, gan)
                st.session_state["aviso"] = "Apuesta cerrada."
                st.rerun()
        st.subheader("Historial")
        if df.empty:
            st.caption("Todavía no hay apuestas registradas.")
        else:
            st.dataframe(df.iloc[::-1], hide_index=True)

# ===== Pestaña 3: Panel =====
with tab3:
    if not sheets_ok:
        st.info("Conecta Google Sheets para ver el panel.")
    else:
        cerr = df[df.Resultado.isin(CERRADAS)]
        pendientes = int((df.Resultado == "Pendiente").sum())
        m1, m2, m3 = st.columns(3)
        m1.metric("Bank actual", f"{bank:,.2f}", f"{bank - inicial:+,.2f}")
        if cerr.empty:
            m2.metric("ROI", "—")
            m3.metric("Acierto", "—")
            st.info("Aún no hay apuestas cerradas. Registra y cierra algunas para ver las gráficas.")
        else:
            apostado = cerr.Monto.sum()
            validas = cerr.Resultado.isin(["Ganada", "Perdida"]).sum()
            ganadas = (cerr.Resultado == "Ganada").sum()
            m2.metric("ROI", f"{cerr.Ganancia.sum() / apostado:.1%}" if apostado else "—")
            m3.metric("Acierto", f"{ganadas / validas:.0%}" if validas else "—")
            st.caption(f"{len(cerr)} apuestas cerradas · {pendientes} pendientes · "
                       f"{apostado:,.2f} apostado en total")

            st.markdown("**Evolución del bank**")
            evol = pd.DataFrame({"Bank": [inicial] + list(inicial + cerr.Ganancia.cumsum())})
            st.line_chart(evol)
            st.markdown("**Resultados**")
            st.bar_chart(cerr.Resultado.value_counts())
            st.markdown("**Ganancia por mercado**")
            st.bar_chart(cerr.groupby("Mercado").Ganancia.sum())
            st.markdown("**Ganancia por liga**")
            st.bar_chart(cerr.groupby("Liga").Ganancia.sum())
