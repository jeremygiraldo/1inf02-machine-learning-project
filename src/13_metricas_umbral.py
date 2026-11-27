"""Entrega final: porcentaje de predicciones con error dentro de 2, 3 y 5 min. NO reevalúa la prueba.

Métrica descriptiva añadida después de la evaluación única: el umbral de 5 min es el de la
convención de puntualidad de TCRP (retrasado > 5 min). En la prueba se calcula con las
predicciones ya guardadas por 08_final.py; en validación cruzada se reentrena el HistGB final
en los mismos 5 pliegues (solo desarrollo).

Uso: python src/13_metricas_umbral.py   (después de 08 --evaluar-prueba)
"""
import importlib
import json

import numpy as np
import pandas as pd
from sklearn.base import clone

reg = importlib.import_module("04_regresion")
fin = importlib.import_module("08_final")
RES = reg.RES
UMBRALES = [2, 3, 5]


def main():
    p = pd.read_csv(RES / "final_prueba_predicciones.csv")
    filas = []
    for m in ["Persistencia", "Ridge", "Random Forest", "HistGB E2", "HistGradientBoosting"]:
        e = (p[m] - p.real).abs()
        filas.append({"conjunto": "prueba", "modelo": m, **{f"pct_err_le_{u}": 100 * (e <= u).mean() for u in UMBRALES}})

    dev, _ = fin.cargar()
    pliegues = reg.pliegues_forward_chaining(dev.fecha)
    conf = json.loads((RES / "final_configuracion.json").read_text())
    X, y = fin.variables(dev, conf["grupos"]), dev[fin.OBJETIVO]
    m = reg.modelos(X.columns)[0]["HistGradientBoosting"].set_params(
        **conf["hiperparametros_regresion"]["HistGradientBoosting"])
    por_pliegue = []
    for tr, va in pliegues:
        e = np.abs(y.iloc[va] - clone(m).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[va]))
        por_pliegue.append([100 * (e <= u).mean() for u in UMBRALES])
    por_pliegue = np.array(por_pliegue)
    filas.append({"conjunto": "validación cruzada", "modelo": "HistGradientBoosting",
                  **{f"pct_err_le_{u}": por_pliegue[:, i].mean() for i, u in enumerate(UMBRALES)},
                  "pct_err_le_5_sd": por_pliegue[:, 2].std(ddof=1)})
    t = pd.DataFrame(filas)
    t.round(2).to_csv(RES / "final_pct_error_umbral.csv", index=False)
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
