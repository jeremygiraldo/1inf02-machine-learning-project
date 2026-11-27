"""Variables en tiempo real para la entrega final: el estado del tramo que el bus aún no recorre.

Para cada vuelta, el instante de decisión t es la hora real estimada de paso por P10
(hora programada + último retraso observado). Solo se usa información anterior a t:

  Grupo "precedentes" (Times.csv, otras vueltas del mismo día, ventana de 30 min):
    prec_ganancia_10_14  media de (d14 - d10) de las vueltas que pasaron P14 en (t-30, t]
    prec_ganancia_14_18  media de (d18 - d14) de las vueltas que pasaron P18 en (t-30, t]
    prec_n_10_14, prec_n_14_18  cuántas vueltas aportan a cada media

  Grupo "gps_adelante" (Monitoring.csv, otros buses, ventana de 15 min):
    gps_vel_10_14, gps_vel_14_18  velocidad media de los buses que en (t-15, t] estaban
    entre P10 y P14 (o entre P14 y P18). Un punto GPS está en el tramo si su bus ya
    había pasado el punto inicial y aún no el final, según sus propias horas reales.

  Grupo "gps_propio" (Monitoring.csv, el mismo bus entre P5 y t):
    gps_vel_propia, gps_detenido_propio  velocidad media y fracción de puntos con
    velocidad <= 2 km/h

Escribe data/variables_tiempo_real.csv (una fila por id de vuelta).
Uso: python src/07_tiempo_real.py   (después de 01)
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
K = 10
VENTANA_PREC = 30
VENTANA_GPS = 15
DETENIDO = 2
MAX_TRAMO = 60

GRUPOS = {
    "precedentes": ["prec_ganancia_10_14", "prec_ganancia_14_18", "prec_n_10_14", "prec_n_14_18"],
    "gps_adelante": ["gps_vel_10_14", "gps_vel_14_18"],
    "gps_propio": ["gps_vel_propia", "gps_detenido_propio"],
}


def horas_reales(df):
    for i in range(1, 19):
        df[f"e{i}"] = df[f"s{i}"] + df[f"d{i}"]
    ultimo = df[[f"d{i}" for i in range(1, K + 1)]].ffill(axis=1).iloc[:, -1].fillna(0)
    df["t_decision"] = df[f"s{K}"] + ultimo
    return df


def ventana(tiempos_ord, valores_ord, t, ancho):
    acum = np.concatenate([[0.0], np.cumsum(valores_ord)])
    lo = np.searchsorted(tiempos_ord, t - ancho, side="right")
    hi = np.searchsorted(tiempos_ord, t, side="right")
    n = hi - lo
    suma = acum[hi] - acum[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, suma / n, np.nan), n


def precedentes(df):
    out = pd.DataFrame(index=df.index)
    for a, b in [(10, 14), (14, 18)]:
        media = np.full(len(df), np.nan)
        cuenta = np.zeros(len(df))
        for fecha, idx in df.groupby("fecha").groups.items():
            dia = df.loc[idx]
            ok = dia[f"d{a}"].notna() & dia[f"d{b}"].notna()
            fin = dia.loc[ok, f"e{b}"].to_numpy()
            gan = (dia.loc[ok, f"d{b}"] - dia.loc[ok, f"d{a}"]).to_numpy()
            orden = np.argsort(fin)
            m, n = ventana(fin[orden], gan[orden], dia.t_decision.to_numpy(), VENTANA_PREC)
            pos = df.index.get_indexer(idx)
            media[pos], cuenta[pos] = m, n
        out[f"prec_ganancia_{a}_{b}"] = media
        out[f"prec_n_{a}_{b}"] = cuenta
    return out


def cargar_gps():
    g = pd.read_csv(RAIZ / "Dataset/Monitoring.csv", dtype={"vehicle_id_id": str},
                    usecols=["date", "time", "speed", "vehicle_id_id"])
    hms = g.time.str.split(":", expand=True).astype(int)
    g["tau"] = hms[0] * 60 + hms[1] + hms[2] / 60
    return g.rename(columns={"date": "fecha", "vehicle_id_id": "vehiculo"})[
        ["fecha", "vehiculo", "tau", "speed"]].sort_values("tau").reset_index(drop=True)


def asignar_tramo(gps, df, a, b):
    """Marca los puntos GPS de buses que ya pasaron Pa y aún no Pb (según su vuelta)."""
    vueltas = df[df[f"d{a}"].notna()][["fecha", "vehiculo", "id", f"e{a}", f"e{b}", f"s{a}", f"s{b}"]]
    vueltas = vueltas.sort_values(f"e{a}")
    m = pd.merge_asof(gps, vueltas, left_on="tau", right_on=f"e{a}", by=["fecha", "vehiculo"],
                      direction="backward")
    fin = m[f"e{b}"].fillna(m[f"e{a}"] + (m[f"s{b}"] - m[f"s{a}"]))
    dentro = m[f"e{a}"].notna() & (m.tau < fin) & (m.tau - m[f"e{a}"] <= MAX_TRAMO)
    return m.loc[dentro, ["fecha", "vehiculo", "tau", "speed"]]


def gps_adelante(df, gps):
    out = pd.DataFrame(index=df.index)
    for a, b in [(10, 14), (14, 18)]:
        tramo = asignar_tramo(gps, df, a, b)
        media = np.full(len(df), np.nan)
        for fecha, idx in df.groupby("fecha").groups.items():
            pts = tramo[tramo.fecha == fecha]
            pos = df.index.get_indexer(idx)
            for veh, sub in df.loc[idx].groupby("vehiculo"):
                otros = pts[pts.vehiculo != veh]
                m, _ = ventana(otros.tau.to_numpy(), otros.speed.to_numpy().astype(float),
                               sub.t_decision.to_numpy(), VENTANA_GPS)
                media[df.index.get_indexer(sub.index)] = m
        out[f"gps_vel_{a}_{b}"] = media
    return out


def gps_propio(df, gps):
    inicio = df[[f"e{i}" for i in range(5, K + 1)]].bfill(axis=1).iloc[:, 0]
    vueltas = df.assign(inicio=inicio)[["id", "fecha", "vehiculo", "inicio", "t_decision"]]
    vueltas = vueltas[vueltas.inicio.notna()].sort_values("inicio")
    m = pd.merge_asof(gps, vueltas, left_on="tau", right_on="inicio", by=["fecha", "vehiculo"],
                      direction="backward")
    m = m[m.id.notna() & (m.tau <= m.t_decision)]
    agg = m.groupby("id").agg(gps_vel_propia=("speed", "mean"),
                              gps_detenido_propio=("speed", lambda v: np.mean(v <= DETENIDO)))
    return df[["id"]].join(agg, on="id").drop(columns="id")


def main():
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    df = horas_reales(df)
    gps = cargar_gps()
    print(f"GPS: {len(gps):,} puntos")
    var = pd.concat([df[["id"]], precedentes(df), gps_adelante(df, gps), gps_propio(df, gps)], axis=1)
    var.to_csv(RAIZ / "data/variables_tiempo_real.csv", index=False)
    print(var.describe().T.round(2).to_string())
    print("\nFaltantes (%):\n", (100 * var.isna().mean()).round(1).to_string())


if __name__ == "__main__":
    main()
