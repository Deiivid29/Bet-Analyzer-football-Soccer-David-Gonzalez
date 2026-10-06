import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HDR = {"User-Agent": "Mozilla/5.0"}

def leer(url):
    return pd.read_csv(url, encoding="latin-1", storage_options=HDR)

# ---------- 1. DESCARGA ----------
hoy = pd.Timestamp.today()
ini = hoy.year if hoy.month >= 7 else hoy.year - 1
temporadas = [f"{a % 100:02d}{(a + 1) % 100:02d}" for a in range(ini - 5, ini + 1)]
ligas_eu = {"E0": "Premier League", "SP1": "La Liga", "I1": "Serie A",
            "D1": "Bundesliga", "F1": "Ligue 1"}
cols = ["Liga", "Temporada", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

partes, fallos = [], 0
for codigo, nombre in ligas_eu.items():
    for t in temporadas:
        try:
            d = leer(f"https://www.football-data.co.uk/mmz4281/{t}/{codigo}.csv")
            d["Liga"], d["Temporada"] = nombre, t
            partes.append(d[cols])
        except Exception as e:
            fallos += 1
            print("FALLÓ", nombre, t, e)
for codigo, nombre in {"MEX": "Liga MX", "USA": "MLS"}.items():
    try:
        d = leer(f"https://www.football-data.co.uk/new/{codigo}.csv")
        d = d.rename(columns={"Home": "HomeTeam", "Away": "AwayTeam",
                              "HG": "FTHG", "AG": "FTAG", "Res": "FTR"})
        d["Liga"], d["Temporada"] = nombre, d["Season"].astype(str)
        partes.append(d[cols])
    except Exception as e:
        fallos += 1
        print("FALLÓ", nombre, e)

partidos = pd.concat(partes, ignore_index=True)
partidos["Date"] = pd.to_datetime(partidos["Date"], dayfirst=True, format="mixed")
partidos = partidos.dropna(subset=["FTR", "Date"]).drop_duplicates(
    subset=["Liga", "Date", "HomeTeam", "AwayTeam"])
print("Último partido:", partidos.Date.max().date(), "| partidos:", len(partidos),
      "| descargas fallidas:", fallos)
if len(partidos) < 17000:
    sys.exit("Se descargaron muy pocos partidos: no se actualiza nada.")

# ---------- 2. VARIABLES ----------
p = partidos.sort_values("Date").reset_index(drop=True)
p["pid"] = p.index
loc = pd.DataFrame({"pid": p.pid, "Liga": p.Liga, "Date": p.Date, "Equipo": p.HomeTeam,
    "GF": p.FTHG, "GC": p.FTAG, "Pts": p.FTR.map({"H": 3, "D": 1, "A": 0}), "Sede": "L"})
vis = pd.DataFrame({"pid": p.pid, "Liga": p.Liga, "Date": p.Date, "Equipo": p.AwayTeam,
    "GF": p.FTAG, "GC": p.FTHG, "Pts": p.FTR.map({"A": 3, "D": 1, "H": 0}), "Sede": "V"})
largo = pd.concat([loc, vis]).sort_values(["Liga", "Equipo", "Date", "pid"])
largo["GD"] = largo.GF - largo.GC
g = largo.groupby(["Liga", "Equipo"])
for c in ["GF", "GC", "Pts"]:
    largo[c + "_5"] = g[c].transform(lambda s: s.shift().rolling(5, min_periods=3).mean())
largo["GD_10"] = g["GD"].transform(lambda s: s.shift().rolling(10, min_periods=5).mean())
largo["Descanso"] = g["Date"].diff().dt.days.clip(upper=14)
largo["PtsSede_5"] = largo.groupby(["Liga", "Equipo", "Sede"])["Pts"].transform(
    lambda s: s.shift().rolling(5, min_periods=3).mean())

f5 = ["GF_5", "GC_5", "Pts_5", "GD_10", "PtsSede_5", "Descanso"]
l = largo[largo.Sede == "L"][["pid"] + f5].rename(columns={c: "L_" + c for c in f5})
v = largo[largo.Sede == "V"][["pid"] + f5].rename(columns={c: "V_" + c for c in f5})
b = p.merge(l, on="pid").merge(v, on="pid")

elo, el, ev = {}, [], []
for r in p.itertuples():
    e_l = elo.get((r.Liga, r.HomeTeam), 1500)
    e_v = elo.get((r.Liga, r.AwayTeam), 1500)
    el.append(e_l)
    ev.append(e_v)
    esp = 1 / (1 + 10 ** (-(e_l + 60 - e_v) / 400))
    real = {"H": 1, "D": 0.5, "A": 0}[r.FTR]
    elo[(r.Liga, r.HomeTeam)] = e_l + 20 * (real - esp)
    elo[(r.Liga, r.AwayTeam)] = e_v - 20 * (real - esp)
b["L_Elo"], b["V_Elo"] = el, ev
b["Dif_Elo"] = b.L_Elo - b.V_Elo

# ---------- 3. ENTRENAMIENTO ----------
vars_forma = ["L_GF_5", "L_GC_5", "L_Pts_5", "V_GF_5", "V_GC_5", "V_Pts_5",
              "L_Elo", "V_Elo", "Dif_Elo", "L_Descanso", "V_Descanso",
              "L_GD_10", "V_GD_10", "L_PtsSede_5", "V_PtsSede_5"]
datos = b.dropna(subset=vars_forma).sort_values("Date").reset_index(drop=True)
X = pd.get_dummies(datos[["Liga"] + vars_forma], columns=["Liga"]).astype(float)
y = datos["FTR"]
columnas = list(X.columns)

def clasificador():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))

def goles():
    return make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-3, max_iter=1000))

# Control de calidad: entrenar con lo antiguo y probar con lo reciente
corte = int(len(datos) * 0.8)
mt = clasificador().fit(X.iloc[:corte], y.iloc[:corte])
ll = log_loss(y.iloc[corte:], mt.predict_proba(X.iloc[corte:]), labels=mt.classes_)
print(f"Log loss de control: {ll:.4f} (esperado alrededor de 1.01)")
if not 0.95 < ll < 1.06:
    sys.exit("El log loss de control es raro: no se actualiza nada.")

modelo = clasificador().fit(X, y)
gl = goles().fit(X, datos.FTHG)
gv = goles().fit(X, datos.FTAG)
real_g = (datos.FTHG + datos.FTAG).mean()
pred_g = (gl.predict(X) + gv.predict(X)).mean()
print(f"Goles por partido: real {real_g:.2f} | modelo {pred_g:.2f}")
if abs(real_g - pred_g) > 0.2:
    sys.exit("El modelo de goles se ve raro: no se actualiza nada.")

# ---------- 4. GUARDAR ----------
p[["Date", "Liga", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].to_csv(
    "historial.csv", index=False)
joblib.dump({"modelo": modelo, "columnas": columnas, "elo": elo,
             "goles_local": gl, "goles_visitante": gv}, "modelo_apuestas.joblib")
print("Listo: entrenado con", len(datos), "partidos")
