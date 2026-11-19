"""Análisis exploratorio (Entregable 2). Solo usa el periodo de DESARROLLO (6-26 ene),
para que ninguna figura que guíe decisiones mire la prueba.

Salidas: figuras/eda_por_punto.pdf, figuras/eda_p18.pdf y resultados/eda_resumen.csv
Uso: python src/02_eda.py   (después de 01_preprocesamiento.py)
"""
from pathlib import Path

import numpy as np
import pandas as pd

from estilo import ANCHO_COL, AQUA, AZUL, NARANJA, TINTA2, coma, plt

RAIZ = Path(__file__).resolve().parents[1]
FIG = RAIZ / "figuras"
FIN_DESARROLLO = "2025-01-26"
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]


def main():
    FIG.mkdir(exist_ok=True)
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    df = df[df.fecha <= FIN_DESARROLLO].copy()
    D = df[[f"d{i}" for i in range(1, 19)]]
    puntos = np.arange(1, 19)
    q = D.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).T
    faltante = 100 * D.isna().mean()

    # Figura 1: perfil del retraso por punto de control + % de pasos no registrados.
    fig, (a, b) = plt.subplots(2, 1, figsize=(ANCHO_COL, 3.3), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1]})
    a.fill_between(puntos, q[0.1], q[0.9], color=AZUL, alpha=0.15, lw=0, label="p10–p90")
    a.fill_between(puntos, q[0.25], q[0.75], color=AZUL, alpha=0.35, lw=0, label="p25–p75")
    a.plot(puntos, q[0.5], color=AZUL, marker="o", ms=3, label="Mediana")
    a.axhline(0, color=TINTA2, lw=0.8)
    a.axvline(10, color=TINTA2, lw=0.8, ls="--")
    a.text(10.2, 11, "P10: punto\nde predicción", fontsize=6.5, color=TINTA2, va="top")
    a.set_ylabel("Retraso (min)")
    a.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3)
    a.yaxis.set_major_locator(plt.MultipleLocator(5))
    b.bar(puntos, faltante, color=AZUL, width=0.7)
    b.set_ylabel("No registrado (%)")
    b.set_xlabel("Punto de control")
    b.set_xticks(puntos, [f"{p}" for p in puntos])
    coma(b)
    fig.align_ylabels()
    fig.savefig(FIG / "eda_por_punto.pdf")
    plt.close(fig)

    # Figura 2: retraso en P18 por hora de salida (según plantilla) y por día de la semana.
    y = df[df.d18.notna()].copy()
    y["hora"] = (y.s1 // 60).astype(int)
    y["plantilla"] = (y.s18 - y.s1).round().astype(int)
    y["dia"] = pd.to_datetime(y.fecha).dt.dayofweek
    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO_COL * 2, 2.3),
                               gridspec_kw={"width_ratios": [1.6, 1]})
    # La plantilla de 168 min solo existe en la prueba (reemplaza a la de 170 desde el 27 ene).
    for plantilla, color in zip([161, 170, 147], [AZUL, NARANJA, AQUA]):
        s = y[y.plantilla == plantilla].groupby("hora").d18.agg(["median", "size"])
        s = s[s["size"] >= 10]  # se omiten horas con menos de 10 vueltas
        n = (y.plantilla == plantilla).sum()
        a.plot(s.index, s["median"], color=color, marker="o", ms=3,
               label=f"{plantilla} min (n = {n:,})".replace(",", " "))
    a.axhline(0, color=TINTA2, lw=0.8)
    a.set_xlabel("Hora programada de salida")
    a.set_ylabel("Retraso mediano en P18 (min)")
    a.legend(title="Plantilla (desarrollo)", title_fontsize=7, loc="upper left")
    a.set_xticks(range(4, 21, 2))
    grupos = [y.loc[y.dia == d, "d18"].to_numpy() for d in range(7)]
    bp = b.boxplot(grupos, widths=0.55, showfliers=False, patch_artist=True,
                   medianprops={"color": "white", "lw": 1.4},
                   whiskerprops={"color": TINTA2}, capprops={"color": TINTA2})
    for caja, d in zip(bp["boxes"], range(7)):
        caja.set(facecolor=NARANJA if d == 5 else AZUL, edgecolor="none")
    b.set_xticks(range(1, 8), [f"{DIAS[d]}\n{len(g)}" for d, g in enumerate(grupos)])
    b.axhline(0, color=TINTA2, lw=0.8)
    b.set_ylabel("Retraso en P18 (min)")
    b.set_xlabel("Día (debajo: n vueltas)")
    fig.tight_layout()
    fig.savefig(FIG / "eda_p18.pdf")
    plt.close(fig)

    # Resumen numérico que respalda el texto del paper.
    resumen = pd.DataFrame({"punto": puntos, "p10": q[0.1].values, "p25": q[0.25].values,
                            "mediana": q[0.5].values, "p75": q[0.75].values, "p90": q[0.9].values,
                            "pct_no_registrado": faltante.values})
    resumen.round(2).to_csv(RAIZ / "resultados/eda_resumen.csv", index=False)
    por_dia = y.groupby("dia").d18.agg(["size", "median", "mean"]).rename(index=dict(enumerate(DIAS)))
    print(f"Desarrollo: {len(df)} vueltas, {len(y)} con P18\n", resumen.round(1).to_string(index=False))
    print(por_dia.round(1).to_string())


if __name__ == "__main__":
    main()
