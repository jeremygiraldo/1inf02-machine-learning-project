"""Entrega final: genera la demo interactiva (un solo HTML, funciona sin conexión).

Para cada vuelta de prueba guarda el horario, los retrasos observados, la predicción del
HistGB entrenado para cada punto de partida k = 5..17 (variables del E2 e hiperparámetros
de 09_curva_k.py, entrenado solo con el desarrollo), la predicción del modelo final en P10
(08_final.py), la clase predicha por el clasificador elegido en validación cruzada y las
variables en tiempo real. La ruta y los puntos de control se dibujan con el GPS.

Uso: python src/11_demo.py   (después de 07, 08 --evaluar-prueba y 09)
Salida: Entrega final/demo/index.html
"""
import ast
import importlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone

from features import OBJETIVO, construir

reg = importlib.import_module("04_regresion")
RAIZ, RES = reg.RAIZ, reg.RES
VUELTA_RUTA = 30844
ANCHO = 1000
SALIDA = RAIZ / "Entrega final/demo/index.html"


def num(v, d=2):
    return None if pd.isna(v) else round(float(v), d)


def predicciones_por_k(dev, test):
    curva = pd.read_csv(RES / "final_curva_k_ajustada.csv")
    curva = curva[curva.modelo == "HistGradientBoosting"].set_index("k")
    preds = {}
    for k in range(5, 18):
        Xd, Xt = construir(dev, k, vuelta_anterior=False), construir(test, k, vuelta_anterior=False)
        m = reg.modelos(Xd.columns)[0]["HistGradientBoosting"].set_params(**ast.literal_eval(curva.loc[k, "parametros"]))
        preds[k] = clone(m).fit(Xd, dev[OBJETIVO]).predict(Xt)
    return preds, curva.MAE.round(2).tolist()


def geometria(df):
    g = pd.read_csv(RAIZ / "Dataset/Monitoring.csv", dtype={"vehicle_id_id": str},
                    usecols=["date", "time", "latitude", "longitude", "vehicle_id_id"])
    hms = g.time.str.split(":", expand=True).astype(int)
    g["tau"] = hms[0] * 60 + hms[1] + hms[2] / 60
    g = g.rename(columns={"date": "fecha", "vehicle_id_id": "vehiculo"}).sort_values("tau")
    pasos = []
    for i in range(1, 19):
        p = df[df[f"d{i}"].notna()][["fecha", "vehiculo"]].assign(cp=i, tau=df[f"s{i}"] + df[f"d{i}"])
        pasos.append(p.dropna())
    pasos = pd.concat(pasos).sort_values("tau")
    j = pd.merge_asof(pasos, g, on="tau", by=["fecha", "vehiculo"], direction="nearest", tolerance=1.0)
    cps = j.groupby("cp")[["latitude", "longitude"]].median()

    v = df[df.id == VUELTA_RUTA].iloc[0]
    tr = g[(g.fecha == v.fecha) & (g.vehiculo == v.vehiculo) & (g.tau >= v.s1 + v.d1) & (g.tau <= v.s18 + v.d18)]
    lat0 = np.deg2rad(cps.latitude.mean())
    todo = pd.concat([tr[["latitude", "longitude"]], cps])
    x0, x1 = todo.longitude.min(), todo.longitude.max()
    y1 = todo.latitude.max()
    esc = ANCHO / ((x1 - x0) * np.cos(lat0))
    proy = lambda la, lo: (round((lo - x0) * np.cos(lat0) * esc, 1), round((y1 - la) * esc, 1))
    ruta = [proy(a, b) for a, b in zip(tr.latitude, tr.longitude)]
    alto = max(p[1] for p in ruta + [proy(a, b) for a, b in zip(cps.latitude, cps.longitude)])
    taus = tr.tau.to_numpy()
    puntos = []
    for i in range(1, 19):
        x, y = proy(cps.loc[i, "latitude"], cps.loc[i, "longitude"])
        idx = int(np.argmin(np.abs(taus - (v[f"s{i}"] + v[f"d{i}"]))))
        puntos.append({"x": x, "y": y, "i": idx})
    return ruta, puntos, ANCHO, round(alto, 1)


def etiquetas(puntos):
    lab = [None] * len(puntos)
    for j, p in enumerate(puntos):
        if lab[j] is not None:
            continue
        grupo = [i for i, q in enumerate(puntos) if abs(q["x"] - p["x"]) < 35 and abs(q["y"] - p["y"]) < 35]
        texto = " · ".join(f"P{i + 1}" for i in grupo)
        lab[j] = [0, -18, texto]
        for i in grupo[1:]:
            lab[i] = [0, 0, ""]
    return lab


def main():
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    tr = pd.read_csv(RAIZ / "data/variables_tiempo_real.csv")
    con = df[df[OBJETIVO].notna()].merge(tr, on="id", how="left")
    dev = con[con.fecha <= reg.FIN_DESARROLLO].reset_index(drop=True)
    test = con[con.fecha >= reg.INICIO_PRUEBA].reset_index(drop=True)

    pk, mae_k = predicciones_por_k(dev, test)
    pf = pd.read_csv(RES / "final_prueba_predicciones.csv").set_index("id")
    cvc = pd.read_csv(RES / "final_cv_clasificacion.csv")
    clf = cvc.loc[cvc.F1_macro_media.idxmax(), "modelo"]
    pc = pd.read_csv(RES / "final_prueba_predicciones_clase.csv").set_index("id")[clf]
    tabla = pd.read_csv(RES / "final_prueba_regresion.csv").set_index("modelo")

    laps = []
    for i, r in test.iterrows():
        laps.append({
            "id": int(r.id), "f": r.fecha, "v": r.vehiculo, "pl": int(round(r.s18 - r.s1)),
            "s": [num(r[f"s{j}"], 1) for j in range(1, 19)],
            "d": [num(r[f"d{j}"]) for j in range(1, 19)],
            "pk": [num(pk[k][i]) for k in range(5, 18)],
            "pf": num(pf.loc[r.id, "HistGradientBoosting"]), "pe2": num(pf.loc[r.id, "HistGB E2"]),
            "cl": int(pc.loc[r.id]),
            "rt": {"g": num(r.prec_ganancia_10_14), "n": int(r.prec_n_10_14), "v": num(r.gps_vel_10_14, 1)},
        })
    ruta, puntos, W, H = geometria(df)
    datos = {"laps": laps, "route": ruta, "cps": puntos, "lab": etiquetas(puntos), "W": W, "H": H,
             "mae": mae_k, "mae_final": float(tabla.loc["HistGradientBoosting", "MAE"]),
             "pie": f"{len(laps)} vueltas de prueba con P18 observado. Predicción por punto: HistGB entrenado "
                    f"para cada k con 6–26 ene; en P10, modelo final con información en tiempo real. "
                    f"Clase: {clf} (elegido por F1 macro en validación cruzada). Umbrales: adelantado < −1 min, "
                    f"retrasado > 5 min."}
    html = (RAIZ / "src/demo_plantilla.html").read_text().replace(
        "/*DATOS*/null", json.dumps(datos, ensure_ascii=False, separators=(",", ":")))
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(html)
    print(f"Demo escrita en {SALIDA} ({len(html) / 1024:.0f} KB, {len(laps)} vueltas)")


if __name__ == "__main__":
    main()
