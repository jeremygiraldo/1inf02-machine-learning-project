"""Análisis de la revisión del Entregable 2. NO reentrena ni reevalúa modelos en la prueba.

  A. Comparación estadística por día (unidad = día, no vuelta) con las predicciones de
     prueba YA guardadas: diferencia de MAE con IC por bootstrap pareado por día y
     prueba de signo sobre los 6 días. Corrige la Wilcoxon por vuelta, que trata como
     independientes vueltas del mismo día y compara medianas, no MAE.
  B. Descomposición del error de prueba por subgrupo (sábado, plantilla nueva de 168 min).
  C. Sesgo de selección: vueltas de desarrollo sin P18 frente a vueltas con P18.
  D. Pasos con |d| > 60 min: hora de reloj y coincidencia con pasos de otras vueltas.
  E. Datos de apoyo: vueltas por hora (línea base), vueltas por vehículo, efecto del ajuste
     de hiperparámetros y tiempos de entrenamiento e inferencia (solo con desarrollo).

Uso: python src/06_revision.py   (después de 01, 04 y 05)
"""
import importlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.base import clone

from features import OBJETIVO, construir

reg = importlib.import_module("04_regresion")
RAIZ, RES = reg.RAIZ, reg.RES
MODELOS = ["Retraso cero", "Persistencia", "Media histórica por hora", "Ridge", "Random Forest",
           "HistGradientBoosting"]
MEJOR = "HistGradientBoosting"


def a_min(hhmmss):
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return h * 60 + m + s / 60


# ------------------------------------------------------------------ A
def comparacion_por_dia(p, rng, n_boot=5000):
    err = p[MODELOS].sub(p.real, axis=0).abs()
    mae_dia = err.groupby(p.fecha).mean()
    dias = mae_dia.index.to_numpy()
    idx = {d: np.flatnonzero(p.fecha.to_numpy() == d) for d in dias}
    filas = []
    for m in MODELOS:
        if m == MEJOR:
            continue
        dif = err[m] - err[MEJOR]  # > 0: el mejor modelo tiene menos error
        boots = []
        for _ in range(n_boot):
            sel = np.concatenate([idx[d] for d in rng.choice(dias, len(dias), replace=True)])
            boots.append(dif.iloc[sel].mean())
        gana = int((mae_dia[m] > mae_dia[MEJOR]).sum())
        filas.append({"modelo": m, "dif_MAE": dif.mean(),
                      "ic_inf": np.quantile(boots, 0.025), "ic_sup": np.quantile(boots, 0.975),
                      "dias_gana_HistGB": gana, "dias": len(dias),
                      "p_signo_bilateral": binomtest(gana, len(dias), 0.5).pvalue})
    t = pd.DataFrame(filas)
    t.round(4).to_csv(RES / "prueba_comparacion_por_dia.csv", index=False)
    return t


# ------------------------------------------------------------------ B
def descomposicion(p, df):
    p = p.merge(df[["id", "s1", "s18"]], on="id", validate="one_to_one")
    p["grupo"] = np.select(
        [pd.to_datetime(p.fecha).dt.dayofweek == 5, (p.s18 - p.s1).round() == 168],
        ["Sábado 1 feb", "Plantilla 168 (lun-vie)"], "Resto (lun-vie)")
    p["err"] = (p[MEJOR] - p.real).abs()
    total = p.err.sum()
    t = p.groupby("grupo").agg(vueltas=("err", "size"), MAE=("err", "mean"), suma=("err", "sum"))
    t["pct_vueltas"] = 100 * t.vueltas / len(p)
    t["pct_error_total"] = 100 * t.suma / total
    t = t.drop(columns="suma")
    t.round(3).to_csv(RES / "prueba_descomposicion_error.csv")
    return t


# ------------------------------------------------------------------ C
def sesgo_seleccion(dev_todo):
    g = dev_todo.assign(con_p18=dev_todo[OBJETIVO].notna(),
                        sabado=pd.to_datetime(dev_todo.fecha).dt.dayofweek == 5,
                        hora=dev_todo.s1 // 60)
    t = g.groupby("con_p18").agg(
        vueltas=("id", "size"), pct_sabado=("sabado", "mean"), hora_media=("hora", "mean"),
        d10_mediana=("d10", "median"), d14_mediana=("d14", "median"), d16_mediana=("d16", "median"),
        d14_p75=("d14", lambda s: s.quantile(.75)), d16_p75=("d16", lambda s: s.quantile(.75)))
    t["pct_sabado"] *= 100
    t.round(2).to_csv(RES / "sesgo_seleccion_p18.csv")
    return t


# ------------------------------------------------------------------ D
def pasos_anomalos(rng):
    T = pd.read_csv(RAIZ / "Dataset/Times.csv", dtype={"vehicle_id_id": str})
    sep = lambda s: s.strip("{}").split(",")
    filas = []
    for r in T.itertuples():
        S = [a_min(x) for x in sep(r.start_time)]
        if any(b < a for a, b in zip(S, S[1:])):
            continue  # horario corrupto (se excluye en 01)
        dia = (pd.Timestamp(r.date) - pd.Timestamp("2025-01-01")).days
        for k, (s, e) in enumerate(zip(S, sep(r.end_time))):
            if e != "00:00:00":
                filas.append((r.id, r.vehicle_id_id, dia, k + 1, a_min(e), a_min(e) - s, S[0]))
    C = pd.DataFrame(filas, columns=["id", "veh", "dia", "punto", "real", "d", "salida"])
    C["t_abs"] = C.dia * 1440 + C.real
    validos = C[C.d.abs() <= 60]
    por_veh = {v: g for v, g in validos.groupby("veh")}

    def distancia(fila, desplazamiento):
        o = por_veh[fila.veh]
        o = o[o.id != fila.id]
        t = fila.dia * 1440 + fila.real + desplazamiento
        return (o.t_abs - t).abs().min()

    anom = C[C.d.abs() > 60]
    # Distancia mínima a un paso válido de otra vuelta del mismo bus (mismo día o siguiente).
    dist = [min(distancia(f, 0), distancia(f, 1440)) for f in anom.itertuples()]
    ref = []
    for f in validos.sample(300, random_state=reg.SEMILLA).itertuples():
        sh = rng.uniform(-600, 600)
        ref.append(min(distancia(f, sh), distancia(f, sh + 1440)))
    neg = anom[anom.d < 0]
    res = {
        "celdas": len(anom), "negativas": int((anom.d < 0).sum()), "positivas": int((anom.d > 0).sum()),
        "negativas_hora_reloj_4a6": int(((neg.real >= 240) & (neg.real < 420)).sum()),
        "negativas_salida_16a20": int(((neg.salida >= 960) & (neg.salida < 1260)).sum()),
        "pct_a_2min_de_otro_paso": 100 * np.mean(np.array(dist) <= 2),
        "pct_a_2min_referencia_azar": 100 * np.mean(np.array(ref) <= 2),
        "puntos_mas_frecuentes": anom.punto.value_counts().head(5).to_dict(),
    }
    (RES / "pasos_anomalos.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=int))
    return res


# ------------------------------------------------------------------ E
def apoyo(dev):
    X = construir(dev, reg.K, vuelta_anterior=False)
    y = dev[OBJETIVO]
    conf = json.loads((RES / "configuracion.json").read_text())
    hgb = reg.modelos(X.columns)[0]["HistGradientBoosting"].set_params(
        **conf["hiperparametros"]["HistGradientBoosting"])
    t0 = time.perf_counter()
    f = clone(hgb).fit(X, y)
    t_fit = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20):
        f.predict(X)
    t_pred = (time.perf_counter() - t0) / 20 / len(X)
    hora = (dev.s1 // 60).value_counts()
    veh = dev.vehiculo.value_counts()
    abl = pd.read_csv(RES / "ablacion_vuelta_anterior.csv")
    defecto = abl[~abl.vuelta_anterior].set_index("modelo").MAE_cv
    ajustado = pd.read_csv(RES / "cv_resultados.csv").set_index("modelo").MAE_media
    res = {
        "vueltas_por_hora_min": int(hora.min()), "horas_con_menos_de_30": int((hora < 30).sum()),
        "vueltas_por_vehiculo_min": int(veh.min()), "vueltas_por_vehiculo_mediana": float(veh.median()),
        "vueltas_por_vehiculo_max": int(veh.max()), "vehiculos": int(veh.size),
        "mae_cv_defecto": defecto.round(3).to_dict(),
        "mae_cv_ajustado": ajustado.loc[defecto.index].round(3).to_dict(),
        "entrenamiento_HistGB_s": round(t_fit, 2), "inferencia_ms_por_vuelta": round(1000 * t_pred, 4),
    }
    (RES / "apoyo_revision.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    return res


def main():
    rng = np.random.default_rng(reg.SEMILLA)
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    p = pd.read_csv(RES / "prueba_predicciones.csv")
    print("A. Diferencia de MAE frente a HistGB (unidad: día):\n",
          comparacion_por_dia(p, rng).round(3).to_string(index=False))
    print("\nB. Descomposición del error de prueba:\n", descomposicion(p, df).round(2))
    dev_todo = df[df.fecha <= reg.FIN_DESARROLLO]
    print("\nC. Sesgo de selección (desarrollo):\n", sesgo_seleccion(dev_todo).round(2).T)
    print("\nD. Pasos con |d| > 60:\n", pasos_anomalos(rng))
    dev = dev_todo[dev_todo[OBJETIVO].notna()].reset_index(drop=True)
    print("\nE. Apoyo:\n", json.dumps(apoyo(dev), indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
