"""Entrega final: figuras del paper a partir de los CSV de resultados (no entrena nada).

  figuras/final_importancia.pdf      importancia por permutación del HistGB final (CV)
  figuras/final_por_dia.pdf          MAE por día de prueba: persistencia, HistGB E2 y HistGB final
  figuras/final_matriz_confusion.pdf matriz de confusión del clasificador elegido en CV (prueba)

Uso: python src/12_figuras_final.py   (después de 08 --evaluar-prueba)
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from estilo import ANCHO_COL, AZUL, NARANJA, AQUA, TINTA, TINTA2, coma, plt

reg = importlib.import_module("04_regresion")
RES, FIG = reg.RES, reg.RAIZ / "figuras"
NOMBRES = {"hora_sin": "Hora de salida (seno)", "hora_cos": "Hora de salida (coseno)",
           "dia_semana": "Día de la semana", "vehiculo": "Vehículo",
           "ultimo_retraso": "Último retraso observado", "plantilla": "Plantilla",
           "restante_prog": "Tiempo programado restante", "vuelta": "N.º de vuelta",
           "variacion": "Variación del retraso", "n_faltantes": "Pasos faltantes",
           "prec_ganancia_10_14": "Ganancia P10→P14 de precedentes", "prec_n_10_14": "N.º de precedentes P10→P14",
           "prec_ganancia_14_18": "Ganancia P14→P18 de precedentes", "prec_n_14_18": "N.º de precedentes P14→P18",
           "gps_vel_10_14": "Velocidad GPS P10→P14", "gps_vel_14_18": "Velocidad GPS P14→P18"}
TIEMPO_REAL = {k for k in NOMBRES if k.startswith(("prec_", "gps_"))}


def importancia():
    t = pd.read_csv(RES / "final_importancia.csv").head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(ANCHO_COL, 2.5))
    colores = [NARANJA if v in TIEMPO_REAL else AZUL for v in t.variable]
    ax.barh([NOMBRES.get(v, f"Retraso en P{v[1:]}") for v in t.variable], t.aumento_MAE, xerr=t.sd_pliegues,
            color=colores, height=0.6, error_kw={"ecolor": TINTA2, "lw": 0.8, "capsize": 2})
    ax.set_xlabel("Aumento del MAE al permutar (min)")
    ax.grid(axis="y", visible=False)
    coma(ax, "x", 1)
    fig.savefig(FIG / "final_importancia.pdf")
    plt.close(fig)


def por_dia():
    d = pd.read_csv(RES / "final_prueba_mae_por_dia.csv")
    etiquetas = ["Lun 27", "Mar 28", "Mié 29", "Jue 30", "Vie 31", "Sáb 1"]
    x = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(ANCHO_COL, 2.2))
    for i, (col, color, nombre) in enumerate([("Persistencia", NARANJA, "Persistencia"),
                                              ("HistGB E2", AQUA, "HistGB E2 (solo horarios)"),
                                              ("HistGradientBoosting", AZUL, "HistGB final (tiempo real)")]):
        ax.bar(x + (i - 1) * 0.27, d[col], width=0.27, color=color, label=nombre)
    ax.set_xticks(x, etiquetas)
    ax.set_ylabel("MAE (min)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left")
    coma(ax)
    fig.savefig(FIG / "final_por_dia.pdf")
    plt.close(fig)


def matriz():
    cv = pd.read_csv(RES / "final_cv_clasificacion.csv")
    elegido = cv.loc[cv.F1_macro_media.idxmax(), "modelo"]
    p = pd.read_csv(RES / "final_prueba_predicciones_clase.csv")
    v = confusion_matrix(p.real, p[elegido], labels=[0, 1, 2])
    pd.DataFrame(v, index=["real Adelantado", "real Puntual", "real Retrasado"],
                 columns=["pred Adelantado", "pred Puntual", "pred Retrasado"]).to_csv(
        RES / "final_prueba_matriz_confusion_elegido_cv.csv")
    print(f"Matriz de confusión de {elegido} (elegido por F1 macro en validación cruzada)")
    pct = v / v.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(ANCHO_COL * 0.8, 2.3))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    clases = ["Adelantado", "Puntual", "Retrasado"]
    ax.set_xticks(range(3), clases)
    ax.set_yticks(range(3), clases)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{v[i, j]}\n{100 * pct[i, j]:.0f} %", ha="center", va="center", fontsize=7,
                    color="white" if pct[i, j] > 0.55 else TINTA)
    fig.savefig(FIG / "final_matriz_confusion.pdf")
    plt.close(fig)


if __name__ == "__main__":
    importancia()
    por_dia()
    matriz()
