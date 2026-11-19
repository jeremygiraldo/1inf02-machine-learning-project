"""Preprocesamiento de Times.csv (Entregable 2).

Convierte los arreglos de 18 puntos de control en columnas, recalcula el retraso
en minutos con segundos (hora real - hora programada) y aplica las reglas de
limpieza fijadas antes de modelar. Escribe:

  data/vueltas_limpias.csv   una fila por vuelta conservada
  resultados/calidad.csv     tabla de calidad (cuántas filas/celdas toca cada regla)

Uso:  python src/01_preprocesamiento.py [--umbral 60]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
N_PUNTOS = 18
SALTO_MIN = 15  # salto entre puntos consecutivos que se marca (no se elimina)


def a_minutos(hhmmss: str) -> float:
    h, m, s = (int(x) for x in hhmmss.split(":"))
    return h * 60 + m + s / 60


def separar(arreglo: str) -> list[str]:
    return arreglo.strip("{}").split(",")


def main(umbral: float) -> None:
    T = pd.read_csv(RAIZ / "Dataset/Times.csv", dtype={"vehicle_id_id": str})
    n_vueltas, n_celdas = len(T), len(T) * N_PUNTOS

    prog = T.start_time.map(separar)
    real = T.end_time.map(separar)
    dif = T.difference_time.map(separar)
    assert (prog.map(len) == N_PUNTOS).all() and (real.map(len) == N_PUNTOS).all() \
        and (dif.map(len) == N_PUNTOS).all(), "algún arreglo no tiene 18 valores"
    assert not T.duplicated(["vehicle_id_id", "date", "lap"]).any(), "vuelta duplicada"

    S = pd.DataFrame([[a_minutos(x) for x in fila] for fila in prog])
    E = pd.DataFrame([[np.nan if x == "00:00:00" else a_minutos(x) for x in fila] for fila in real])
    Dorig = pd.DataFrame([[np.nan if x == "-" else float(x) for x in fila] for fila in dif])
    # Los pasos no registrados aparecen igual en ambas columnas ("-" y 00:00:00).
    assert (E.isna() == Dorig.isna()).all().all(), "faltantes de end_time y difference_time no coinciden"

    D = E - S  # retraso en minutos con segundos; positivo = tarde
    # difference_time es la parte entera de D: se verifica en vez de suponerlo.
    obs = Dorig.notna()
    coincide = (np.trunc(D[obs]) == Dorig[obs]).sum().sum() / obs.sum().sum()

    filas = []

    def registrar(regla, unidad, n, total, tratamiento):
        filas.append(dict(regla=regla, unidad=unidad, afectados=int(n), total=int(total),
                          porcentaje=round(100 * n / total, 2), tratamiento=tratamiento))

    registrar("Paso no registrado ('-' / 00:00:00)", "celda", E.isna().sum().sum(), n_celdas,
              "NaN + indicador; nunca se imputa el objetivo")

    # R1. Horario programado no creciente: arreglo corrupto -> excluir vuelta.
    r1 = ~(S.diff(axis=1).iloc[:, 1:] >= 0).all(axis=1)
    registrar("Horario programado no creciente", "vuelta", r1.sum(), n_vueltas, "excluir vuelta")

    # R2. Paso asignado a otra vuelta (|retraso| > umbral) -> NaN en esa celda.
    r2 = (D.abs() > umbral) & ~r1.values[:, None]
    registrar(f"Paso asignado a otra vuelta (|d| > {umbral:g} min)", "celda", r2.sum().sum(),
              n_celdas, "marcar como NaN")
    # Tramos negativos antes de aplicar R2 (para documentar que R2 los explica).
    neg_antes = ((E.ffill(axis=1).diff(axis=1).iloc[:, 1:] < 0).any(axis=1) & ~r1).sum()
    D, E = D.mask(r2), E.mask(r2)

    # R3. Tiempo de tramo negativo (entre pasos observados) -> excluir vuelta.
    tramo = E.ffill(axis=1).diff(axis=1).iloc[:, 1:]
    r3 = (tramo < 0).any(axis=1) & ~r1
    registrar(f"Tiempo de tramo negativo (antes de R2: {neg_antes} vueltas)", "vuelta", r3.sum(),
              n_vueltas, "excluir vuelta")

    # R4. Salto de retraso > 15 min entre puntos observados consecutivos -> solo se marca.
    salto = D.ffill(axis=1).diff(axis=1).abs().iloc[:, 1:]
    r4 = (salto > SALTO_MIN).any(axis=1) & ~r1 & ~r3
    registrar(f"Salto de retraso > {SALTO_MIN} min", "vuelta", r4.sum(), n_vueltas,
              "conservar con indicador")

    conservar = ~r1 & ~r3
    out = pd.DataFrame({
        "id": T.id, "fecha": T.date, "vuelta": T.lap, "vehiculo": T.vehicle_id_id,
        "salto_grande": r4.astype(int),
    })
    for k in range(N_PUNTOS):
        out[f"s{k + 1}"] = S[k].round(4)
    for k in range(N_PUNTOS):
        out[f"d{k + 1}"] = D[k].round(4)
    out = out[conservar].sort_values(["vehiculo", "fecha", "vuelta"]).reset_index(drop=True)

    registrar("Vueltas conservadas", "vuelta", len(out), n_vueltas, "-")
    registrar("Vueltas conservadas con P18 observado", "vuelta", out.d18.notna().sum(), n_vueltas,
              "muestra para predecir P18")

    # Invariantes de la salida.
    assert out.id.is_unique
    assert (out[[f"s{k + 1}" for k in range(N_PUNTOS)]].diff(axis=1).iloc[:, 1:] >= 0).all().all()
    assert (out[[f"d{k + 1}" for k in range(N_PUNTOS)]].abs().max() <= umbral).all()

    (RAIZ / "data").mkdir(exist_ok=True)
    (RAIZ / "resultados").mkdir(exist_ok=True)
    out.to_csv(RAIZ / "data/vueltas_limpias.csv", index=False)
    calidad = pd.DataFrame(filas)
    calidad.to_csv(RAIZ / "resultados/calidad.csv", index=False)

    print(f"difference_time == trunc(real - programada): {coincide:.4%} de las celdas observadas")
    print(calidad.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--umbral", type=float, default=60.0,
                   help="|retraso| (min) por encima del cual el paso se considera de otra vuelta")
    main(p.parse_args().umbral)
