"""Regresión del retraso en P18 desde P10 (Entregable 2, nivel 1).

Protocolo (fijado antes de ver la prueba):
  * Desarrollo = 6-26 ene 2025; prueba = 27 ene - 1 feb 2025 (por fecha de salida).
  * Validación cruzada forward-chaining por días dentro de desarrollo:
    primeros 11 días de entrenamiento inicial y 5 pliegues de 2 días de validación.
  * Ablación de la variable "retraso de la vuelta anterior": se incluye si baja el
    MAE medio de validación cruzada en al menos 2 de los 3 modelos (config. por defecto).
  * Hiperparámetros: búsqueda aleatoria con esa misma validación cruzada (MAE).
  * Prueba: modelo reentrenado con todo desarrollo, evaluado UNA sola vez
    (solo con --evaluar-prueba; deja un registro y se niega a repetir sin --forzar).

Uso:
  python src/04_regresion.py                  # ablación + ajuste (sin tocar la prueba)
  python src/04_regresion.py --evaluar-prueba # además, evaluación final única
"""
import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, wilcoxon
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAS, OBJETIVO, construir

RAIZ = Path(__file__).resolve().parents[1]
RES = RAIZ / "resultados"
SEMILLA = 42
K = 10                      # punto de control desde el que se predice
FIN_DESARROLLO = "2025-01-26"
INICIO_PRUEBA = "2025-01-27"
N_ITER = 40                 # configuraciones por búsqueda aleatoria
N_BOOT = 2000               # remuestreos por día para los IC


# ---------------------------------------------------------------- líneas base
class RetrasoCero(BaseEstimator, RegressorMixin):
    """Supone que el bus llega exactamente a la hora programada."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X))


class Persistencia(BaseEstimator, RegressorMixin):
    """El último retraso observado hasta k se mantiene hasta P18."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["ultimo_retraso"].fillna(0).to_numpy()


class MediaHistoricaHora(BaseEstimator, RegressorMixin):
    """Media del retraso en P18 por hora programada de salida (ajustada en entrenamiento)."""
    def fit(self, X, y):
        hora = self._hora(X)
        self.medias_ = pd.Series(np.asarray(y)).groupby(hora.to_numpy()).mean()
        self.global_ = float(np.mean(y))
        return self

    def predict(self, X):
        return self._hora(X).map(self.medias_).fillna(self.global_).to_numpy()

    @staticmethod
    def _hora(X):
        ang = np.arctan2(X["hora_sin"], X["hora_cos"]) % (2 * np.pi)
        return (ang / (2 * np.pi) * 24 + 1e-6).astype(int)


# ---------------------------------------------------------------- modelos
def modelos(columnas):
    numericas = [c for c in columnas if c not in CATEGORICAS]
    ridge = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]),
        Ridge())
    rf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEMILLA))
    hgb = HistGradientBoostingRegressor(categorical_features=CATEGORICAS, early_stopping=False,
                                        random_state=SEMILLA)
    espacios = {
        "Ridge": {"ridge__alpha": loguniform(1e-3, 1e3)},
        "Random Forest": {
            "randomforestregressor__max_depth": [None, 6, 10, 16],
            "randomforestregressor__min_samples_leaf": [1, 2, 5, 10, 20],
            "randomforestregressor__max_features": [0.3, 0.5, 0.8, 1.0],
        },
        "HistGradientBoosting": {
            "learning_rate": loguniform(0.01, 0.2),
            "max_iter": [100, 200, 400, 800],
            "max_leaf_nodes": [7, 15, 31, 63],
            "min_samples_leaf": [5, 10, 20, 40],
            "l2_regularization": loguniform(1e-3, 10),
        },
    }
    return {"Ridge": ridge, "Random Forest": rf, "HistGradientBoosting": hgb}, espacios


LINEAS_BASE = {"Retraso cero": RetrasoCero(), "Persistencia": Persistencia(),
               "Media histórica por hora": MediaHistoricaHora()}


# ---------------------------------------------------------------- utilidades
def pliegues_forward_chaining(fechas: pd.Series, n_inicial=11, tam=2):
    dias = np.sort(fechas.unique())
    assert len(dias) == n_inicial + 5 * tam, f"se esperaban 21 días de desarrollo, hay {len(dias)}"
    salida = []
    for i in range(n_inicial, len(dias), tam):
        tr = np.flatnonzero(fechas.isin(dias[:i]))
        va = np.flatnonzero(fechas.isin(dias[i:i + tam]))
        salida.append((tr, va))
    return salida


def metricas(y, p):
    e = np.abs(np.asarray(y) - np.asarray(p))
    return {"MAE": e.mean(), "RMSE": np.sqrt(mean_squared_error(y, p)), "R2": r2_score(y, p),
            "pct_err_le_2": 100 * np.mean(e <= 2)}


def cv_por_pliegue(modelo, X, y, pliegues):
    filas = []
    for i, (tr, va) in enumerate(pliegues):
        m = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        filas.append({"pliegue": i + 1, **metricas(y.iloc[va], m.predict(X.iloc[va]))})
    return pd.DataFrame(filas)


def bootstrap_por_dia(y, p, fechas, rng):
    dias = fechas.unique()
    grupos = {d: np.flatnonzero(fechas.to_numpy() == d) for d in dias}
    muestras = []
    for _ in range(N_BOOT):
        idx = np.concatenate([grupos[d] for d in rng.choice(dias, len(dias), replace=True)])
        muestras.append(metricas(y.iloc[idx], p[idx]))
    m = pd.DataFrame(muestras)
    return m.quantile(0.025), m.quantile(0.975)


def commit_actual():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=RAIZ, text=True).strip()
    except Exception:
        return "desconocido"


# ---------------------------------------------------------------- flujo
def main(evaluar_prueba: bool, forzar: bool):
    RES.mkdir(exist_ok=True)
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    df = df[df[OBJETIVO].notna()].reset_index(drop=True)   # nunca se imputa el objetivo
    dev = df[df.fecha <= FIN_DESARROLLO].reset_index(drop=True)
    test = df[df.fecha >= INICIO_PRUEBA].reset_index(drop=True)
    assert len(dev) + len(test) == len(df)
    pliegues = pliegues_forward_chaining(dev.fecha)
    print(f"Desarrollo: {len(dev)} vueltas, {dev.fecha.nunique()} días | Prueba: {len(test)} vueltas, "
          f"{test.fecha.nunique()} días | Pliegues: " +
          ", ".join(f"{len(tr)}/{len(va)}" for tr, va in pliegues))

    # 1. Ablación de la vuelta anterior (configuración por defecto, solo validación cruzada).
    ablacion = []
    for usar in (False, True):
        X = construir(dev, K, vuelta_anterior=usar)
        mods, _ = modelos(X.columns)
        for nombre, m in mods.items():
            r = cv_por_pliegue(m, X, dev[OBJETIVO], pliegues)
            ablacion.append({"vuelta_anterior": usar, "modelo": nombre,
                             "MAE_cv": r.MAE.mean(), "MAE_cv_sd": r.MAE.std()})
    ablacion = pd.DataFrame(ablacion)
    piv = ablacion.pivot(index="modelo", columns="vuelta_anterior", values="MAE_cv")
    usar_anterior = bool((piv[True] < piv[False]).sum() >= 2)
    ablacion.round(4).to_csv(RES / "ablacion_vuelta_anterior.csv", index=False)
    print("\nAblación (MAE CV):\n", piv.round(3), f"\n-> usar vuelta anterior: {usar_anterior}")

    X_dev = construir(dev, K, vuelta_anterior=usar_anterior)
    y_dev = dev[OBJETIVO]
    mods, espacios = modelos(X_dev.columns)

    # 2. Ajuste de hiperparámetros con la validación cruzada forward-chaining.
    finales, top5, elegidos = dict(LINEAS_BASE), [], {}
    for nombre, m in mods.items():
        n_iter = 30 if nombre == "Ridge" else N_ITER
        bus = RandomizedSearchCV(m, espacios[nombre], n_iter=n_iter, cv=pliegues,
                                 scoring="neg_mean_absolute_error", random_state=SEMILLA,
                                 n_jobs=1 if nombre == "Random Forest" else -1, refit=False)
        bus.fit(X_dev, y_dev)
        cr = pd.DataFrame(bus.cv_results_).sort_values("rank_test_score").head(5)
        for _, fila in cr.iterrows():
            top5.append({"modelo": nombre, "rango": int(fila.rank_test_score),
                         "MAE_cv": -fila.mean_test_score, "MAE_cv_sd": fila.std_test_score,
                         "parametros": json.dumps({k.split("__")[-1]: (round(v, 5) if isinstance(v, float) else v)
                                                   for k, v in fila.params.items()})})
        elegidos[nombre] = bus.best_params_
        finales[nombre] = clone(m).set_params(**bus.best_params_)
        print(f"{nombre}: MAE CV {-bus.best_score_:.3f} con {bus.best_params_}")
    pd.DataFrame(top5).round(4).to_csv(RES / "hiperparametros_top5.csv", index=False)

    # 3. Desempeño en validación cruzada de todos los modelos (configuración elegida).
    cv = []
    for nombre, m in finales.items():
        r = cv_por_pliegue(m, X_dev, y_dev, pliegues)
        cv.append({"modelo": nombre, **{f"{c}_media": r[c].mean() for c in ["MAE", "RMSE", "R2", "pct_err_le_2"]},
                   **{f"{c}_sd": r[c].std() for c in ["MAE", "RMSE", "R2", "pct_err_le_2"]}})
    cv = pd.DataFrame(cv)
    cv.round(4).to_csv(RES / "cv_resultados.csv", index=False)
    print("\nValidación cruzada (media de 5 pliegues):\n",
          cv[["modelo", "MAE_media", "MAE_sd", "RMSE_media", "R2_media", "pct_err_le_2_media"]].round(3).to_string(index=False))

    with open(RES / "configuracion.json", "w") as f:
        json.dump({"k": K, "semilla": SEMILLA, "vuelta_anterior": usar_anterior,
                   "variables": list(X_dev.columns),
                   "hiperparametros": {n: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                                           for k, v in p.items()} for n, p in elegidos.items()}},
                  f, indent=2, ensure_ascii=False, default=str)

    if not evaluar_prueba:
        print("\nPrueba NO evaluada (usar --evaluar-prueba cuando el protocolo esté congelado).")
        return

    # 4. Evaluación final única en la prueba.
    registro = RES / "PRUEBA_EVALUADA.txt"
    if registro.exists() and not forzar:
        raise SystemExit(f"La prueba ya se evaluó (ver {registro}). Usa --forzar solo si es imprescindible.")
    X_test = construir(test, K, vuelta_anterior=usar_anterior)
    y_test = test[OBJETIVO]
    rng = np.random.default_rng(SEMILLA)
    filas, preds = [], pd.DataFrame({"id": test.id, "fecha": test.fecha, "real": y_test})
    for nombre, m in finales.items():
        p = clone(m).fit(X_dev, y_dev).predict(X_test)
        preds[nombre] = p
        lo, hi = bootstrap_por_dia(y_test, p, test.fecha, rng)
        filas.append({"modelo": nombre, **metricas(y_test, p),
                      **{f"{c}_ic_inf": lo[c] for c in lo.index}, **{f"{c}_ic_sup": hi[c] for c in hi.index}})
    tabla = pd.DataFrame(filas)
    mejor = tabla.loc[tabla.MAE.idxmin(), "modelo"]
    err = preds[list(finales)].sub(preds.real, axis=0).abs()
    tabla["p_wilcoxon_vs_mejor"] = [np.nan if n == mejor else wilcoxon(err[mejor], err[n]).pvalue
                                    for n in tabla.modelo]
    tabla.round(4).to_csv(RES / "prueba_resultados.csv", index=False)
    preds.round(4).to_csv(RES / "prueba_predicciones.csv", index=False)
    por_dia = err.groupby(preds.fecha).mean()
    por_dia.insert(0, "n_vueltas", preds.groupby("fecha").size())
    por_dia.round(3).to_csv(RES / "prueba_mae_por_dia.csv")
    registro.write_text(f"Prueba evaluada {dt.datetime.now().isoformat(timespec='seconds')} "
                        f"(commit {commit_actual()})\n")
    print("\nPRUEBA (evaluación única):\n",
          tabla[["modelo", "MAE", "MAE_ic_inf", "MAE_ic_sup", "RMSE", "R2", "R2_ic_inf", "R2_ic_sup",
                 "pct_err_le_2", "p_wilcoxon_vs_mejor"]].round(3).to_string(index=False))
    print("\nMAE por día:\n", por_dia.round(2).to_string())


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--evaluar-prueba", action="store_true")
    a.add_argument("--forzar", action="store_true")
    args = a.parse_args()
    main(args.evaluar_prueba, args.forzar)
