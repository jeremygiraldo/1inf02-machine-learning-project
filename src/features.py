"""Variables causales para predecir el retraso en P18 desde el punto de control k.

Solo se usa información disponible cuando el bus pasa por k: retrasos d1..dk,
el horario programado (conocido de antemano), calendario, vehículo y el retraso
final de la vuelta anterior del mismo bus (ya terminada antes de que salga esta;
se verificó que ninguna vuelta sale antes de que termine la anterior).
"""
import numpy as np
import pandas as pd

OBJETIVO = "d18"
CATEGORICAS = ["vehiculo", "dia_semana"]


def construir(df: pd.DataFrame, k: int, vuelta_anterior: bool = True) -> pd.DataFrame:
    d = df[[f"d{i}" for i in range(1, k + 1)]]
    X = d.copy()
    ultimo = d.ffill(axis=1).iloc[:, -1]
    X["ultimo_retraso"] = ultimo                      # retraso más reciente observado
    X["variacion"] = ultimo - d.ffill(axis=1).iloc[:, -2] if k > 1 else 0.0
    X["n_faltantes"] = d.isna().sum(axis=1)
    salida = df["s1"]                                 # hora programada de salida (min)
    X["hora_sin"] = np.sin(2 * np.pi * salida / 1440)
    X["hora_cos"] = np.cos(2 * np.pi * salida / 1440)
    X["plantilla"] = (df["s18"] - df["s1"]).round()   # duración programada: 147/161/168/170
    X["restante_prog"] = df["s18"] - df[f"s{k}"]
    X["vuelta"] = df["vuelta"]
    X["dia_semana"] = pd.to_datetime(df["fecha"]).dt.dayofweek
    X["vehiculo"] = df["vehiculo"].astype(int)
    if vuelta_anterior:
        X["retraso_vuelta_anterior"] = retraso_vuelta_anterior(df)
    return X


def retraso_vuelta_anterior(df: pd.DataFrame) -> pd.Series:
    """Retraso en P18 de la vuelta inmediatamente anterior del mismo bus y día.

    NaN si es la primera vuelta del día, si hay un hueco en la numeración
    (no sabemos si la vuelta no se hizo o no se registró) o si su P18 falta.
    """
    orden = df.sort_values(["vehiculo", "fecha", "vuelta"])
    g = orden.groupby(["vehiculo", "fecha"])
    previo = g[OBJETIVO].shift()
    consecutiva = g["vuelta"].diff() == 1
    return previo.where(consecutiva).reindex(df.index)
