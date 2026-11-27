"""Entrega final: curva de error por punto de partida con hiperparámetros ajustados para cada k.

En el E2 la curva k = 5..17 reutilizó los hiperparámetros de k = 10. Aquí, para cada k,
se repite la búsqueda aleatoria de Ridge (15 configuraciones) y HistGB (20) con los mismos
5 pliegues y las variables del E2. Solo validación cruzada; la prueba no se usa.

Uso: python src/09_curva_k.py   (después de 01 y 04)
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

from estilo import ANCHO_COL, AZUL, AQUA, NARANJA, TINTA2, coma, plt
from features import OBJETIVO, construir

reg = importlib.import_module("04_regresion")
RAIZ, RES, FIG = reg.RAIZ, reg.RES, reg.RAIZ / "figuras"
N_ITER = {"Ridge": 15, "HistGradientBoosting": 20}


def main():
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    dev = df[df[OBJETIVO].notna() & (df.fecha <= reg.FIN_DESARROLLO)].reset_index(drop=True)
    pliegues = reg.pliegues_forward_chaining(dev.fecha)
    y = dev[OBJETIVO]
    filas = []
    for k in range(5, 18):
        X = construir(dev, k, vuelta_anterior=False)
        mods, espacios = reg.modelos(X.columns)
        for nombre in N_ITER:
            bus = RandomizedSearchCV(mods[nombre], espacios[nombre], n_iter=N_ITER[nombre], cv=pliegues,
                                     scoring="neg_mean_absolute_error", random_state=reg.SEMILLA,
                                     n_jobs=-1, refit=False).fit(X, y)
            r = reg.cv_por_pliegue(mods[nombre].set_params(**bus.best_params_), X, y, pliegues)
            filas.append({"k": k, "modelo": nombre, "MAE": r.MAE.mean(), "MAE_sd": r.MAE.std(),
                          "R2": r.R2.mean(), "R2_sd": r.R2.std(), "pct_err_le_2": r.pct_err_le_2.mean(),
                          "parametros": str({p.split("__")[-1]: (float(round(v, 4)) if isinstance(v, float) else v)
                                             for p, v in bus.best_params_.items()})})
        print(f"k={k} listo")
    curva = pd.DataFrame(filas)
    curva.round(4).to_csv(RES / "final_curva_k_ajustada.csv", index=False)

    e2 = pd.read_csv(RES / "curva_por_k_cv.csv")
    comp = curva.merge(e2[["k", "modelo", "MAE", "R2"]], on=["k", "modelo"], suffixes=("", "_hiper_k10"))
    comp["mejora_MAE"] = comp.MAE_hiper_k10 - comp.MAE
    comp.round(4).to_csv(RES / "final_curva_k_comparacion.csv", index=False)
    print(comp[["k", "modelo", "MAE", "MAE_hiper_k10", "R2", "R2_hiper_k10"]].round(3).to_string(index=False))

    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO_COL * 2, 2.3))
    pers = e2[e2.modelo == "Persistencia"]
    for ax, met in [(a, "MAE"), (b, "R2")]:
        ax.plot(pers.k, pers[met], color=NARANJA, marker="o", ms=3, label="Persistencia")
        for nombre, color in [("Ridge", AQUA), ("HistGradientBoosting", AZUL)]:
            c = curva[curva.modelo == nombre]
            ax.plot(c.k, c[met], color=color, marker="o", ms=3, label=f"{nombre} (ajustado por k)")
            ax.fill_between(c.k, c[met] - c[f"{met}_sd"], c[met] + c[f"{met}_sd"], color=color, alpha=0.12, lw=0)
            v = e2[e2.modelo == nombre]
            ax.plot(v.k, v[met], color=color, ls="--", lw=0.9, alpha=0.8, label=f"{nombre} (hiperpar. de k=10)")
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
    a.legend(loc="upper right", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(FIG / "final_curva_k.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
