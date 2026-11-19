# 1inf02-machine-learning-project

Predicción del retraso de buses en los puntos de control de una ruta de Cusco
(Aprendizaje de Máquina, PUCP 2026-2).

## Entregable 2: cómo reproducir

```bash
pip install -r requirements.txt
python src/01_preprocesamiento.py          # limpieza -> data/vueltas_limpias.csv, resultados/calidad.csv
python src/04_regresion.py                 # ablación + ajuste con validación cruzada (no toca la prueba)
python src/02_eda.py                       # figuras del EDA (solo periodo de desarrollo)
python src/04_regresion.py --evaluar-prueba  # evaluación final única en la prueba
python src/05_analisis.py                  # curva por k, sensibilidad del umbral, importancia (CV) y desglose de la prueba
python src/06_revision.py                  # comparación por día, descomposición del error, sesgo de selección, pasos anómalos, tiempos
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
| `src/02_eda.py` | Figuras del análisis exploratorio (solo desarrollo) |
| `src/05_analisis.py` | Curva de error por punto de partida, sensibilidad del umbral e importancia por permutación (validación cruzada); desglose de la prueba con las predicciones guardadas, sin reevaluar |
| `src/06_revision.py` | Análisis de la revisión: comparación entre modelos con el día como unidad, descomposición del error, sesgo de selección y verificación de los pasos anómalos (sin reevaluar la prueba) |
| `src/estilo.py` | Estilo común de las figuras |
| `src/features.py` | Variables causales hasta el punto de control k |
| `src/04_regresion.py` | Líneas base, Ridge, Random Forest, HistGradientBoosting; validación cruzada *forward-chaining*, búsqueda de hiperparámetros, prueba única con IC por día y Wilcoxon |
| `resultados/` | Todas las tablas del paper (CSV) y la configuración elegida |
| `figuras/` | Figuras del paper (PDF) |
| `scripts/` | Exploración inicial del E1 (rutas fijas a `/mnt/...`; no se usan en el E2) |
