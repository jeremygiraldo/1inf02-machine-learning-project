"""Entrega final: agrupamiento de perfiles de retraso (aprendizaje no supervisado).

Cada vuelta del desarrollo con los 18 retrasos observados es un perfil de 18 valores (min).
  * K-means (k = 2..8, 20 inicializaciones) y mezcla gaussiana (GMM, covarianza completa).
  * k se elige con el coeficiente de silueta (K-means) y el BIC (GMM); se reporta también
    Davies-Bouldin. Además del k elegido se describe k = 3, la solución del E1, porque el
    k elegido puede separar solo el grupo más extremo. PCA a 2 componentes solo para visualizar.
  * Cada grupo se describe por su perfil medio, hora de salida, plantilla y día de la semana.
  * Las vueltas de prueba completas se asignan al centroide más cercano (sin reajustar) para
    ver si los grupos se mantienen y en cuáles se concentra el error de la regresión final.

Uso: python src/10_agrupamiento.py   (después de 01 y 08 --evaluar-prueba)
"""
import importlib

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

from estilo import ANCHO_COL, AMARILLO, AQUA, AZUL, NARANJA, TINTA2, coma, plt

reg = importlib.import_module("04_regresion")
RAIZ, RES, FIG, SEMILLA = reg.RAIZ, reg.RES, reg.RAIZ / "figuras", reg.SEMILLA
COLS = [f"d{i}" for i in range(1, 19)]
COLORES = [AZUL, NARANJA, AQUA, AMARILLO, TINTA2, "#8a5cc2", "#c25c8a", "#5c8ac2"]


def describir(df, etiquetas, nombres):
    d = df.assign(grupo=etiquetas, hora=df.s1 / 60, plantilla=(df.s18 - df.s1).round(),
                  sabado=pd.to_datetime(df.fecha).dt.dayofweek == 5)
    t = d.groupby("grupo").agg(n=("id", "size"), d10_medio=("d10", "mean"), d14_medio=("d14", "mean"),
                               d18_medio=("d18", "mean"), hora_media=("hora", "mean"),
                               pct_sabado=("sabado", "mean"),
                               plantilla_moda=("plantilla", lambda v: v.mode().iloc[0]))
    t["pct_vueltas"] = 100 * t.n / t.n.sum()
    t["pct_sabado"] *= 100
    t.insert(0, "nombre", [nombres[g] for g in t.index])
    return t


def nombrar(centros):
    final = centros[:, -1]
    orden = np.argsort(final)
    nombres = {}
    for pos, g in enumerate(orden):
        if final[g] < -5:
            nombres[g] = "Adelantado"
        elif final[g] > 8:
            nombres[g] = "Retraso alto" if pos == len(orden) - 1 else "Retraso moderado"
        else:
            nombres[g] = "Normal"
    vistos = {}
    for g in sorted(nombres, key=lambda g: final[g]):
        n = nombres[g]
        vistos[n] = vistos.get(n, 0) + 1
        if list(nombres.values()).count(n) > 1:
            nombres[g] = f"{n} {vistos[n]}"
    return nombres


def main():
    df = pd.read_csv(RAIZ / "data/vueltas_limpias.csv", dtype={"vehiculo": str})
    completos = df[df[COLS].notna().all(axis=1)].reset_index(drop=True)
    dev = completos[completos.fecha <= reg.FIN_DESARROLLO].reset_index(drop=True)
    test = completos[completos.fecha >= reg.INICIO_PRUEBA].reset_index(drop=True)
    X = dev[COLS].to_numpy()
    print(f"Perfiles completos: desarrollo {len(dev)}, prueba {len(test)}")

    filas = []
    for k in range(2, 9):
        km = KMeans(k, n_init=20, random_state=SEMILLA).fit(X)
        gm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEMILLA).fit(X)
        eg = gm.predict(X)
        filas.append({"k": k, "silueta_kmeans": silhouette_score(X, km.labels_),
                      "davies_bouldin_kmeans": davies_bouldin_score(X, km.labels_),
                      "silueta_gmm": silhouette_score(X, eg) if len(set(eg)) > 1 else np.nan,
                      "bic_gmm": gm.bic(X)})
    sel = pd.DataFrame(filas)
    sel.round(4).to_csv(RES / "final_agrupamiento_seleccion.csv", index=False)
    print(sel.round(3).to_string(index=False))
    k_sel = int(sel.loc[sel.silueta_kmeans.idxmax(), "k"])
    p = pd.read_csv(RES / "final_prueba_predicciones.csv")
    soluciones = {}
    for k in sorted({k_sel, 3}):
        km = KMeans(k, n_init=20, random_state=SEMILLA).fit(X)
        nombres = nombrar(km.cluster_centers_)
        tabla = describir(dev, km.labels_, nombres)
        tabla.round(3).to_csv(RES / f"final_agrupamiento_grupos_k{k}.csv")
        print(f"\nk = {k}{' (elegido por silueta)' if k == k_sel else ''}:\n", tabla.round(2).to_string())
        et_test = km.predict(test[COLS].to_numpy())
        reparto = pd.DataFrame({"dev_pct": pd.Series(km.labels_).value_counts(normalize=True) * 100,
                                "prueba_pct": pd.Series(et_test).value_counts(normalize=True) * 100}).fillna(0)
        reparto.index = [nombres[g] for g in reparto.index]
        t2 = test[["id"]].assign(grupo=[nombres[g] for g in et_test]).merge(p, on="id")
        for m in ["Persistencia", "HistGB E2", "HistGradientBoosting"]:
            reparto[f"MAE_{m}"] = (t2[m] - t2.real).abs().groupby(t2.grupo).mean()
        reparto["n_prueba_con_P18"] = t2.groupby("grupo").size()
        reparto.round(3).to_csv(RES / f"final_agrupamiento_prueba_k{k}.csv")
        print("Reparto y error de la regresión por grupo (prueba):\n", reparto.round(2).to_string())
        soluciones[k] = (km, nombres)
    km, nombres = soluciones[3]

    pca = PCA(2, random_state=SEMILLA).fit(X)
    Z = pca.transform(X)
    fig, (a, b) = plt.subplots(1, 2, figsize=(ANCHO_COL * 2, 2.4), gridspec_kw={"width_ratios": [1.25, 1]})
    for g in sorted(nombres, key=lambda g: km.cluster_centers_[g, -1]):
        n = (km.labels_ == g).sum()
        a.plot(range(1, 19), km.cluster_centers_[g], color=COLORES[g], marker="o", ms=2.5,
               label=f"{nombres[g]} ({100 * n / len(X):.0f} %)")
        b.scatter(Z[km.labels_ == g, 0], Z[km.labels_ == g, 1], s=3, alpha=0.5, lw=0, color=COLORES[g])
    a.axhline(0, color=TINTA2, lw=0.6)
    a.set_xticks(range(1, 19))
    a.set_xlabel("Punto de control")
    a.set_ylabel("Retraso medio (min)")
    a.legend(loc="upper left", fontsize=6)
    var = pca.explained_variance_ratio_ * 100
    b.set_xlabel(f"CP1 ({var[0]:.0f} % de la varianza)".replace(".", ","))
    b.set_ylabel(f"CP2 ({var[1]:.0f} %)")
    coma(a)
    fig.tight_layout()
    fig.savefig(FIG / "final_agrupamiento.pdf")
    plt.close(fig)
    pd.DataFrame({"componente": ["CP1", "CP2"], "varianza_explicada": pca.explained_variance_ratio_}).round(4).to_csv(
        RES / "final_agrupamiento_pca.csv", index=False)


if __name__ == "__main__":
    main()
