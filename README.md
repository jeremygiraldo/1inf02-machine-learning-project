# 1inf02-machine-learning-project

Predicción del retraso de buses en los puntos de control de una ruta de Cusco
(Aprendizaje de Máquina, PUCP 2026-2).

## Entregable 2: cómo reproducir

```bash
pip install -r requirements.txt
python src/01_preprocesamiento.py          # limpieza -> data/vueltas_limpias.csv, resultados/calidad.csv
python src/04_regresion.py                 # ablación + ajuste con validación cruzada (no toca la prueba)
python src/04_regresion.py --evaluar-prueba  # evaluación final única en la prueba
```

- Datos: `Dataset/Times.csv` (versionado). `Monitoring.csv` (GPS) no está en el repositorio y no se usa en el E2.
- La prueba (27 ene – 1 feb 2025) ya se evaluó una vez (ver `resultados/PRUEBA_EVALUADA.txt`).
  El script se niega a evaluarla de nuevo salvo con `--forzar`.
- Semilla: 42. Versiones exactas en `requirements.txt` (Python 3.13).
- Paper: `paper-entregable-2.tex` → `paper-entregable-2.pdf` (`latexmk -pdf paper-entregable-2.tex`).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/01_preprocesamiento.py` | Reglas de limpieza con aserciones y tabla de calidad |
| `src/features.py` | Variables causales hasta el punto de control k |
| `src/04_regresion.py` | Líneas base, Ridge, Random Forest, HistGradientBoosting; validación cruzada *forward-chaining*, búsqueda de hiperparámetros, prueba única con IC por día y Wilcoxon |
| `resultados/` | Todas las tablas del paper (CSV) y la configuración elegida |
| `scripts/` | Exploración inicial del E1 (rutas fijas a `/mnt/...`; no se usan en el E2) |
