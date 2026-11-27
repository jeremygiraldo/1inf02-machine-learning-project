"""Entrega final: regresión con variables en tiempo real y clasificación de puntualidad.

Protocolo (fijado antes de evaluar la prueba; misma partición y pliegues que el E2):
  A. Ablación por grupo de variables de 07_tiempo_real.py (precedentes, gps_adelante,
     gps_propio) sobre las 20 variables del E2, con los hiperparámetros del E2.
     Regla del E2: un grupo entra si baja el MAE de validación cruzada en al menos
     2 de los 3 modelos.
  B. Con las variables elegidas, nueva búsqueda aleatoria de hiperparámetros (mismos
     espacios, 40 configuraciones por ensamble y 30 para Ridge) y tabla de validación cruzada.
  C. Importancia por permutación del HistGB final en los pliegues de validación.
  D. Clasificación de puntualidad en P18 desde P10 con la convención de TCRP:
     adelantado (d18 < -1), puntual (-1 <= d18 <= 5), retrasado (d18 > 5).
     Métrica principal: F1 macro; complementarias: exactitud balanceada y exactitud.
  E. Prueba: solo con --evaluar-prueba, UNA vez para los modelos nuevos. Antes de evaluar
     se registran las huellas SHA-256 del código y los datos en
     resultados/PRUEBA_FINAL_EVALUADA.txt; el script se niega a repetir sin --forzar.
     Los modelos del E2 no se reevalúan: se comparan con sus predicciones ya guardadas.

Uso:
  python src/08_final.py                  # A-D solo con validación cruzada
  python src/08_final.py --evaluar-prueba # además, evaluación única de la prueba
"""
import argparse
import datetime as dt
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, loguniform
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CATEGORICAS, OBJETIVO, construir

reg = importlib.import_module("04_regresion")
tr_mod = importlib.import_module("07_tiempo_real")
RAIZ, RES, SEMILLA, K = reg.RAIZ, reg.RES, reg.SEMILLA, reg.K
CLASES = ["Adelantado", "Puntual", "Retrasado"]
LIM_ADELANTO, LIM_RETRASO = -1.0, 5.0
N_BOOT = 5000


def clase(d):
    d = np.asarray(d, dtype=float)
    return np.where(d < LIM_ADELANTO, 0, np.where(d > LIM_RETRASO, 2, 1))


def cargar():
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    var = pd.read_csv(RAIZ / "data/variables_tiempo_real.csv")
    df = df.merge(var, on="id", how="left", validate="one_to_one")
    df = df[df[OBJETIVO].notna()].reset_index(drop=True)
    dev = df[df.fecha <= reg.FIN_DESARROLLO].reset_index(drop=True)
    test = df[df.fecha >= reg.INICIO_PRUEBA].reset_index(drop=True)
    return dev, test


def variables(df, grupos):
    X = construir(df, K, vuelta_anterior=False)
    for g in grupos:
        for c in tr_mod.GRUPOS[g]:
            X[c] = df[c].to_numpy()
    return X


def modelos_e2(columnas):
    conf = json.loads((RES / "configuracion.json").read_text())
    mods, espacios = reg.modelos(columnas)
    return {n: m.set_params(**conf["hiperparametros"][n]) for n, m in mods.items()}, espacios


# ------------------------------------------------------------------ A
def ablacion(dev, pliegues):
    y = dev[OBJETIVO]
    base = {n: reg.cv_por_pliegue(m, variables(dev, []), y, pliegues).MAE.mean()
            for n, m in modelos_e2(variables(dev, []).columns)[0].items()}
    filas = [{"grupo": "(ninguno, E2)", "modelo": n, "MAE_cv": v} for n, v in base.items()]
    elegidos = []
    for g in tr_mod.GRUPOS:
        X = variables(dev, [g])
        mejora = 0
        for n, m in modelos_e2(X.columns)[0].items():
            mae = reg.cv_por_pliegue(m, X, y, pliegues).MAE.mean()
            filas.append({"grupo": g, "modelo": n, "MAE_cv": mae})
            mejora += mae < base[n]
        if mejora >= 2:
            elegidos.append(g)
        print(f"Grupo {g}: baja el MAE en {mejora}/3 modelos")
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(RES / "final_ablacion_grupos.csv", index=False)
    return elegidos, tabla


# ------------------------------------------------------------------ B
def ajustar(dev, pliegues, grupos):
    X, y = variables(dev, grupos), dev[OBJETIVO]
    mods, espacios = reg.modelos(X.columns)
    finales, elegidos = {}, {}
    for nombre, m in mods.items():
        bus = RandomizedSearchCV(m, espacios[nombre], n_iter=30 if nombre == "Ridge" else reg.N_ITER,
                                 cv=pliegues, scoring="neg_mean_absolute_error", random_state=SEMILLA,
                                 n_jobs=1 if nombre == "Random Forest" else -1, refit=False).fit(X, y)
        elegidos[nombre] = bus.best_params_
        finales[nombre] = clone(m).set_params(**bus.best_params_)
        print(f"{nombre}: MAE CV {-bus.best_score_:.3f} con {bus.best_params_}")
    return finales, elegidos


def tabla_cv(dev, pliegues, finales, grupos):
    y = dev[OBJETIVO]
    X_e2, X = variables(dev, []), variables(dev, grupos)
    e2 = modelos_e2(X_e2.columns)[0]["HistGradientBoosting"]
    filas = []
    for nombre, m, XX in ([(n, b, X) for n, b in reg.LINEAS_BASE.items()] +
                          [("HistGB E2 (sin tiempo real)", e2, X_e2)] +
                          [(n, m, X) for n, m in finales.items()]):
        r = reg.cv_por_pliegue(m, XX, y, pliegues)
        filas.append({"modelo": nombre, **{f"{c}_media": r[c].mean() for c in ["MAE", "RMSE", "R2", "pct_err_le_2"]},
                      **{f"{c}_sd": r[c].std() for c in ["MAE", "R2"]}, "R2_max_pliegue": r.R2.max()})
    t = pd.DataFrame(filas)
    t.round(4).to_csv(RES / "final_cv_regresion.csv", index=False)
    return t


# ------------------------------------------------------------------ C
def importancia(dev, pliegues, modelo, grupos):
    X, y = variables(dev, grupos), dev[OBJETIVO]
    por_pliegue = []
    for tr, va in pliegues:
        f = clone(modelo).fit(X.iloc[tr], y.iloc[tr])
        r = permutation_importance(f, X.iloc[va], y.iloc[va], n_repeats=10,
                                   scoring="neg_mean_absolute_error", random_state=SEMILLA, n_jobs=-1)
        por_pliegue.append(r.importances_mean)
    imp = pd.DataFrame(por_pliegue, columns=X.columns)
    t = pd.DataFrame({"variable": X.columns, "aumento_MAE": imp.mean().values,
                      "sd_pliegues": imp.std().values}).sort_values("aumento_MAE", ascending=False)
    t.round(4).to_csv(RES / "final_importancia.csv", index=False)
    return t


# ------------------------------------------------------------------ D
class PersistenciaClase(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        self.classes_ = np.array([0, 1, 2])
        return self

    def predict(self, X):
        return clase(X["ultimo_retraso"].fillna(0))


class RegresionUmbralizada(BaseEstimator, ClassifierMixin):
    def __init__(self, regresor=None):
        self.regresor = regresor

    def fit(self, X, y_reg):
        self.classes_ = np.array([0, 1, 2])
        self.regresor_ = clone(self.regresor).fit(X, y_reg)
        return self

    def predict(self, X):
        return clase(self.regresor_.predict(X))


def clasificadores(columnas):
    numericas = [c for c in columnas if c not in CATEGORICAS]
    logreg = make_pipeline(
        ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                                  StandardScaler()), numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
        ]),
        LogisticRegression(class_weight="balanced", max_iter=2000))
    rf = make_pipeline(SimpleImputer(strategy="median", add_indicator=True),
                       RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1,
                                              random_state=SEMILLA))
    hgb = HistGradientBoostingClassifier(categorical_features=CATEGORICAS, class_weight="balanced",
                                         early_stopping=False, random_state=SEMILLA)
    espacios = {
        "Regresión logística": {"logisticregression__C": loguniform(1e-3, 1e2)},
        "Random Forest": {
            "randomforestclassifier__max_depth": [None, 6, 10, 16],
            "randomforestclassifier__min_samples_leaf": [1, 2, 5, 10, 20],
            "randomforestclassifier__max_features": [0.3, 0.5, 0.8, 1.0],
        },
        "HistGradientBoosting": {
            "learning_rate": loguniform(0.01, 0.2),
            "max_iter": [100, 200, 400, 800],
            "max_leaf_nodes": [7, 15, 31, 63],
            "min_samples_leaf": [5, 10, 20, 40],
            "l2_regularization": loguniform(1e-3, 10),
        },
    }
    return {"Regresión logística": logreg, "Random Forest": rf, "HistGradientBoosting": hgb}, espacios


def metricas_clase(y, p):
    return {"F1_macro": f1_score(y, p, average="macro"), "exactitud_balanceada": balanced_accuracy_score(y, p),
            "exactitud": accuracy_score(y, p)}


def clasificacion_cv(dev, pliegues, grupos, regresor_final):
    X, yc, yr = variables(dev, grupos), pd.Series(clase(dev[OBJETIVO])), dev[OBJETIVO]
    mods, espacios = clasificadores(X.columns)
    finales = {"Clase mayoritaria": DummyClassifier(strategy="most_frequent"),
               "Persistencia": PersistenciaClase()}
    elegidos = {}
    for nombre, m in mods.items():
        n_iter = 20 if nombre == "Regresión logística" else 30
        bus = RandomizedSearchCV(m, espacios[nombre], n_iter=n_iter, cv=pliegues, scoring="f1_macro",
                                 random_state=SEMILLA, n_jobs=1 if nombre == "Random Forest" else -1,
                                 refit=False).fit(X, yc)
        elegidos[nombre] = bus.best_params_
        finales[nombre] = clone(m).set_params(**bus.best_params_)
        print(f"Clasificación {nombre}: F1 macro CV {bus.best_score_:.3f}")
    finales["HistGB regresión + umbrales"] = RegresionUmbralizada(regresor_final)
    filas = []
    for nombre, m in finales.items():
        res = []
        for tr, va in pliegues:
            objetivo = yr if isinstance(m, RegresionUmbralizada) else yc
            f = clone(m).fit(X.iloc[tr], objetivo.iloc[tr])
            res.append(metricas_clase(yc.iloc[va], f.predict(X.iloc[va])))
        r = pd.DataFrame(res)
        filas.append({"modelo": nombre, **{f"{c}_media": r[c].mean() for c in r.columns},
                      "F1_macro_sd": r.F1_macro.std()})
    t = pd.DataFrame(filas)
    t.round(4).to_csv(RES / "final_cv_clasificacion.csv", index=False)
    reparto = pd.Series(yc).value_counts(normalize=True).sort_index()
    print("Reparto de clases (desarrollo):", dict(zip(CLASES, reparto.round(3))))
    return finales, elegidos, t


# ------------------------------------------------------------------ E
def huella(rutas):
    h = hashlib.sha256()
    for r in rutas:
        h.update(Path(r).read_bytes())
    return h.hexdigest()[:16]


def bootstrap_dia(fechas, fn, rng, n=N_BOOT):
    dias = np.unique(fechas)
    grupos = {d: np.flatnonzero(fechas == d) for d in dias}
    vals = [fn(np.concatenate([grupos[d] for d in rng.choice(dias, len(dias), replace=True)]))
            for _ in range(n)]
    return np.quantile(vals, [0.025, 0.975])


def comparar_por_dia(err_a, err_b, fechas, rng):
    dif_dia = pd.Series(err_a - err_b).groupby(fechas).mean()
    lo, hi = bootstrap_dia(fechas, lambda i: np.mean(err_a[i]) - np.mean(err_b[i]), rng)
    gana = int((dif_dia > 0).sum())
    return {"dMAE": float(np.mean(err_a) - np.mean(err_b)), "ic_inf": lo, "ic_sup": hi,
            "dias_gana_b": gana, "p_signo": binomtest(gana, len(dif_dia), 0.5).pvalue}


def evaluar_prueba(dev, test, grupos, reg_finales, clf_finales, forzar):
    registro = RES / "PRUEBA_FINAL_EVALUADA.txt"
    if registro.exists() and not forzar:
        raise SystemExit(f"La prueba de la entrega final ya se evaluó (ver {registro}).")
    fuentes = [RAIZ / "src/features.py", RAIZ / "src/04_regresion.py", RAIZ / "src/07_tiempo_real.py",
               Path(__file__), RAIZ / "data/vueltas_limpias.csv", RAIZ / "data/variables_tiempo_real.csv"]
    registro.write_text(f"Prueba final evaluada {dt.datetime.now().isoformat(timespec='seconds')}\n"
                        f"huella SHA-256 (código + datos): {huella(fuentes)}\n"
                        f"grupos: {', '.join(grupos)}\n")
    rng = np.random.default_rng(SEMILLA)
    X_dev, X_test = variables(dev, grupos), variables(test, grupos)
    y_dev, y_test = dev[OBJETIVO], test[OBJETIVO]
    fechas = test.fecha.to_numpy()
    e2 = pd.read_csv(RES / "prueba_predicciones.csv").set_index("id").loc[test.id]

    preds = pd.DataFrame({"id": test.id, "fecha": test.fecha, "real": y_test,
                          "HistGB E2": e2["HistGradientBoosting"].to_numpy(),
                          "Persistencia": e2["Persistencia"].to_numpy()})
    filas = []
    for nombre, m in reg_finales.items():
        preds[nombre] = clone(m).fit(X_dev, y_dev).predict(X_test)
    for nombre in ["Persistencia", "HistGB E2"] + list(reg_finales):
        p = preds[nombre].to_numpy()
        mt = reg.metricas(y_test, p)
        lo, hi = bootstrap_dia(fechas, lambda i: np.mean(np.abs(y_test.to_numpy()[i] - p[i])), rng, 2000)
        lo2, hi2 = bootstrap_dia(fechas, lambda i: reg.r2_score(y_test.to_numpy()[i], p[i]), rng, 2000)
        filas.append({"modelo": nombre, **mt, "MAE_ic_inf": lo, "MAE_ic_sup": hi, "R2_ic_inf": lo2, "R2_ic_sup": hi2})
    tabla = pd.DataFrame(filas)
    tabla.round(4).to_csv(RES / "final_prueba_regresion.csv", index=False)
    preds.round(4).to_csv(RES / "final_prueba_predicciones.csv", index=False)

    err = preds.drop(columns=["id", "fecha"]).sub(preds.real, axis=0).abs().drop(columns="real")
    comp = []
    mejor = "HistGradientBoosting"
    for otro in [c for c in err.columns if c != mejor]:
        comp.append({"modelo": otro, **comparar_por_dia(err[otro].to_numpy(), err[mejor].to_numpy(), fechas, rng)})
    pd.DataFrame(comp).round(4).to_csv(RES / "final_prueba_comparacion_por_dia.csv", index=False)
    por_dia = err.groupby(preds.fecha).mean()
    por_dia.insert(0, "n_vueltas", preds.groupby("fecha").size())
    por_dia.round(3).to_csv(RES / "final_prueba_mae_por_dia.csv")

    yc_dev, yc_test = clase(y_dev), clase(y_test)
    filas_c, pred_c = [], pd.DataFrame({"id": test.id, "fecha": test.fecha, "real": yc_test})
    for nombre, m in clf_finales.items():
        objetivo = y_dev if isinstance(m, RegresionUmbralizada) else yc_dev
        p = clone(m).fit(X_dev, objetivo).predict(X_test)
        pred_c[nombre] = p
        lo, hi = bootstrap_dia(fechas, lambda i: f1_score(yc_test[i], p[i], average="macro"), rng, 2000)
        filas_c.append({"modelo": nombre, **metricas_clase(yc_test, p), "F1_ic_inf": lo, "F1_ic_sup": hi})
    tabla_c = pd.DataFrame(filas_c)
    tabla_c.round(4).to_csv(RES / "final_prueba_clasificacion.csv", index=False)
    pred_c.to_csv(RES / "final_prueba_predicciones_clase.csv", index=False)
    mejor_c = tabla_c.loc[tabla_c.F1_macro.idxmax(), "modelo"]
    cm = pd.DataFrame(confusion_matrix(yc_test, pred_c[mejor_c], labels=[0, 1, 2]),
                      index=[f"real {c}" for c in CLASES], columns=[f"pred {c}" for c in CLASES])
    cm.to_csv(RES / "final_prueba_matriz_confusion.csv")
    return tabla, pd.DataFrame(comp), por_dia, tabla_c, mejor_c, cm


def main(evaluar, forzar):
    dev, test = cargar()
    pliegues = reg.pliegues_forward_chaining(dev.fecha)
    grupos, abl = ablacion(dev, pliegues)
    print("\nA. Ablación (MAE CV):\n", abl.pivot(index="grupo", columns="modelo", values="MAE_cv").round(3),
          f"\n-> grupos elegidos: {grupos}")
    reg_finales, hip = ajustar(dev, pliegues, grupos)
    cv = tabla_cv(dev, pliegues, reg_finales, grupos)
    print("\nB. Validación cruzada:\n", cv.round(3).to_string(index=False))
    imp = importancia(dev, pliegues, reg_finales["HistGradientBoosting"], grupos)
    print("\nC. Importancia (top 12):\n", imp.head(12).round(3).to_string(index=False))
    clf_finales, hip_c, cvc = clasificacion_cv(dev, pliegues, grupos, reg_finales["HistGradientBoosting"])
    print("\nD. Clasificación (CV):\n", cvc.round(3).to_string(index=False))
    with open(RES / "final_configuracion.json", "w") as f:
        json.dump({"k": K, "semilla": SEMILLA, "grupos": grupos,
                   "variables": list(variables(dev, grupos).columns),
                   "umbrales_clase": {"adelantado": f"d18 < {LIM_ADELANTO}", "retrasado": f"d18 > {LIM_RETRASO}"},
                   "hiperparametros_regresion": hip, "hiperparametros_clasificacion": hip_c},
                  f, indent=2, ensure_ascii=False, default=lambda v: float(v) if isinstance(v, np.floating) else str(v))
    if not evaluar:
        print("\nPrueba NO evaluada.")
        return
    tabla, comp, por_dia, tabla_c, mejor_c, cm = evaluar_prueba(dev, test, grupos, reg_finales, clf_finales, forzar)
    print("\nE. PRUEBA regresión:\n", tabla.round(3).to_string(index=False))
    print("\nComparación por día con el HistGB final:\n", comp.round(3).to_string(index=False))
    print("\nMAE por día:\n", por_dia.round(2).to_string())
    print("\nPRUEBA clasificación:\n", tabla_c.round(3).to_string(index=False))
    print(f"\nMatriz de confusión ({mejor_c}):\n", cm.to_string())


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--evaluar-prueba", action="store_true")
    a.add_argument("--forzar", action="store_true")
    args = a.parse_args()
    main(args.evaluar_prueba, args.forzar)
