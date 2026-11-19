"""Análisis complementarios del Entregable 2 (nivel 2). NO vuelve a evaluar la prueba.

  A. Curva de error según el punto de partida k = 5..17 (validación cruzada).
     Se reutilizan los hiperparámetros elegidos para k = 10.
  B. Sensibilidad del umbral de "paso con hora incompatible" (60 vs 30 min),
     comparando sobre las mismas vueltas de validación (validación cruzada).
  C. Importancia por permutación de HistGradientBoosting en los pliegues de validación.
  D. Diagnóstico de la prueba a partir de las predicciones YA guardadas por
     04_regresion.py (resultados/prueba_predicciones.csv): no se reentrena ni se
     reevalúa nada; solo se desglosa el error de la evaluación única.

Uso: python src/05_analisis.py   (después de 01 y 04)
"""
import importlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance

from estilo import ANCHO_COL, AZUL, NARANJA, AQUA, TINTA2, coma, plt
from features import OBJETIVO, construir

reg = importlib.import_module("04_regresion")
RAIZ, RES, FIG = reg.RAIZ, reg.RES, reg.RAIZ / "figuras"
SEMILLA = reg.SEMILLA


def cargar(ruta="data/vueltas_limpias.csv"):
    df = pd.read_csv(RAIZ / ruta, dtype={"vehiculo": str})
    df = df[df[OBJETIVO].notna()]
    return df[df.fecha <= reg.FIN_DESARROLLO].reset_index(drop=True)


def modelos_elegidos(columnas):
    """Ridge y HistGB con los hiperparámetros elegidos en 04 para k = 10."""
    conf = json.loads((RES / "configuracion.json").read_text())
    mods, _ = reg.modelos(columnas)
    return {"Persistencia": reg.Persistencia(),
            "Ridge": mods["Ridge"].set_params(**conf["hiperparametros"]["Ridge"]),
            "HistGradientBoosting": mods["HistGradientBoosting"].set_params(
                **conf["hiperparametros"]["HistGradientBoosting"])}


# ------------------------------------------------------------------ A
def curva_por_k(dev, pliegues):
    filas = []
    for k in range(5, 18):
        X = construir(dev, k, vuelta_anterior=False)
        for nombre, m in modelos_elegidos(X.columns).items():
            r = reg.cv_por_pliegue(m, X, dev[OBJETIVO], pliegues)
            filas.append({"k": k, "modelo": nombre, "MAE": r.MAE.mean(), "MAE_sd": r.MAE.std(),
                          "R2": r.R2.mean(), "R2_sd": r.R2.std(),
                          "pct_err_le_2": r.pct_err_le_2.mean()})
        print(f"k={k} listo")
    curva = pd.DataFrame(filas)
    curva.round(4).to_csv(RES / "curva_por_k_cv.csv", index=False)

    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO_COL * 2, 2.3))
    for (nombre, color) in [("Persistencia", NARANJA), ("Ridge", AQUA), ("HistGradientBoosting", AZUL)]:
        c = curva[curva.modelo == nombre]
        for ax, met in [(a, "MAE"), (b, "R2")]:
            ax.plot(c.k, c[met], color=color, marker="o", ms=3, label=nombre)
            ax.fill_between(c.k, c[met] - c[f"{met}_sd"], c[met] + c[f"{met}_sd"], color=color,
                            alpha=0.12, lw=0)
    b.axhline(0.8, color=TINTA2, lw=0.8, ls="--")
    b.text(5, 0.81, "meta $R^2$ = 0,80", fontsize=6.5, color=TINTA2, va="bottom")
    a.set_ylabel("MAE (min)")
    b.set_ylabel("$R^2$")
    b.set_ylim(-0.1, 1)
    for ax in (a, b):
        ax.set_xlabel("Punto de control desde el que se predice (k)")
        ax.set_xticks(range(5, 18))
        ax.axvline(10, color=TINTA2, lw=0.6, ls=":")
    coma(b, decimales=1)
    a.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "curva_por_k.pdf")
    plt.close(fig)
    return curva


# ------------------------------------------------------------------ B
def sensibilidad_umbral(dev60, pliegues60):
    subprocess.run([sys.executable, str(RAIZ / "src/01_preprocesamiento.py"), "--umbral", "30",
                    "--salida", "data/vueltas_limpias_u30.csv",
                    "--tabla", "resultados/calidad_umbral30.csv"], check=True, capture_output=True)
    dev30 = cargar("data/vueltas_limpias_u30.csv")
    pliegues30 = reg.pliegues_forward_chaining(dev30.fecha)
    comunes = set(dev30.id)  # vueltas cuyo objetivo es válido con ambos umbrales
    filas = []
    for umbral, dev, pl in [(60, dev60, pliegues60), (30, dev30, pliegues30)]:
        X = construir(dev, reg.K, vuelta_anterior=False)
        m = modelos_elegidos(X.columns)["HistGradientBoosting"]
        maes = []
        for tr, va in pl:
            va = va[dev.id.iloc[va].isin(comunes).to_numpy()]
            p = clone(m).fit(X.iloc[tr], dev[OBJETIVO].iloc[tr]).predict(X.iloc[va])
            maes.append(np.mean(np.abs(dev[OBJETIVO].iloc[va] - p)))
        filas.append({"umbral_min": umbral, "vueltas_desarrollo": len(dev),
                      "celdas_marcadas_NaN": int(pd.read_csv(
                          RES / ("calidad.csv" if umbral == 60 else "calidad_umbral30.csv")).afectados.iloc[2]),
                      "MAE_cv_vueltas_comunes": np.mean(maes), "MAE_cv_sd": np.std(maes, ddof=1)})
    s = pd.DataFrame(filas)
    s.round(4).to_csv(RES / "sensibilidad_umbral.csv", index=False)
    (RAIZ / "data/vueltas_limpias_u30.csv").unlink()
    return s


# ------------------------------------------------------------------ C
def importancia(dev, pliegues):
    X = construir(dev, reg.K, vuelta_anterior=False)
    m = modelos_elegidos(X.columns)["HistGradientBoosting"]
    por_pliegue = []
    for tr, va in pliegues:
        f = clone(m).fit(X.iloc[tr], dev[OBJETIVO].iloc[tr])
        r = permutation_importance(f, X.iloc[va], dev[OBJETIVO].iloc[va], n_repeats=10,
                                   scoring="neg_mean_absolute_error", random_state=SEMILLA, n_jobs=-1)
        por_pliegue.append(r.importances_mean)
    imp = pd.DataFrame(por_pliegue, columns=X.columns)
    tabla = pd.DataFrame({"variable": X.columns, "aumento_MAE": imp.mean().values,
                          "sd_pliegues": imp.std().values}).sort_values("aumento_MAE", ascending=False)
    tabla.round(4).to_csv(RES / "importancia_permutacion.csv", index=False)

    top = tabla.head(10).iloc[::-1]
    nombres = {"hora_sin": "Hora de salida (seno)", "hora_cos": "Hora de salida (coseno)",
               "dia_semana": "Día de la semana", "vehiculo": "Vehículo",
               "ultimo_retraso": "Último retraso observado", "plantilla": "Plantilla",
               "restante_prog": "Tiempo programado restante", "vuelta": "N.º de vuelta",
               "variacion": "Variación del retraso", "n_faltantes": "Pasos faltantes"}
    etiquetas = [nombres.get(v, f"Retraso en P{v[1:]}") for v in top.variable]
    fig, ax = plt.subplots(figsize=(ANCHO_COL, 2.4))
    ax.barh(etiquetas, top.aumento_MAE, xerr=top.sd_pliegues, color=AZUL, height=0.6,
            error_kw={"ecolor": TINTA2, "lw": 0.8, "capsize": 2})
    ax.set_xlabel("Aumento del MAE al permutar (min)")
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_locator(plt.MultipleLocator(0.4))
    coma(ax, "x", 1)
    fig.savefig(FIG / "importancia.pdf")
    plt.close(fig)
    return tabla


# ------------------------------------------------------------------ D
def diagnostico_prueba():
    p = pd.read_csv(RES / "prueba_predicciones.csv")
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    p = p.merge(df[["id", "s1", "s18"]], on="id", how="left", validate="one_to_one")
    p["plantilla"] = (p.s18 - p.s1).round().astype(int)
    p["sabado"] = pd.to_datetime(p.fecha).dt.dayofweek == 5
    p["err"] = (p.HistGradientBoosting - p.real).abs()
    p["err_pers"] = (p.Persistencia - p.real).abs()
    plantillas_dev = set((df[df.fecha <= reg.FIN_DESARROLLO].eval("s18 - s1")).round().astype(int))
    por_plantilla = p.groupby("plantilla").agg(n=("err", "size"), MAE_HistGB=("err", "mean"),
                                               MAE_persistencia=("err_pers", "mean"),
                                               retraso_mediano=("real", "median"))
    por_plantilla["vista_en_desarrollo"] = [t in plantillas_dev for t in por_plantilla.index]
    por_plantilla.round(3).to_csv(RES / "prueba_mae_por_plantilla.csv")

    fig, ax = plt.subplots(figsize=(ANCHO_COL, 2.6))
    for sab, color, etiqueta in [(False, AZUL, "Lun–Vie"), (True, NARANJA, "Sábado 1 feb")]:
        s = p[p.sabado == sab]
        ax.scatter(s.real, s.HistGradientBoosting, s=6, color=color, alpha=0.55, lw=0,
                   label=f"{etiqueta} (n = {len(s)})")
    lim = [-30, 42]
    ax.plot(lim, lim, color=TINTA2, lw=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Retraso real en P18 (min)")
    ax.set_ylabel("Predicción HistGB (min)")
    ax.legend(loc="upper left", markerscale=2)
    fig.savefig(FIG / "prueba_real_vs_pred.pdf")
    plt.close(fig)
    return por_plantilla


def main():
    FIG.mkdir(exist_ok=True)
    dev = cargar()
    pliegues = reg.pliegues_forward_chaining(dev.fecha)
    curva = curva_por_k(dev, pliegues)
    print("\nA. Curva por k (CV):\n", curva.pivot(index="k", columns="modelo", values=["MAE", "R2"]).round(3))
    print("\nB. Sensibilidad del umbral:\n", sensibilidad_umbral(dev, pliegues).round(3).to_string(index=False))
    print("\nC. Importancia por permutación (top 10):\n", importancia(dev, pliegues).head(10).round(3).to_string(index=False))
    print("\nD. Prueba por plantilla (predicciones ya guardadas):\n", diagnostico_prueba().round(2))


if __name__ == "__main__":
    main()
