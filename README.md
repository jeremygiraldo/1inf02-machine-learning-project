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
- Paper y entrega: todo en la carpeta `Entregable 2/` (PDF, `.tex` con sus figuras, notebook con todo el código y requirements). Ver `Entregable 2/README.md`, que incluye cómo compilar el `.tex`.

## Entrega final: cómo reproducir

Requiere además `Dataset/Monitoring.csv` (GPS, no versionado).

```bash
python src/07_tiempo_real.py           # variables en tiempo real (otras vueltas + GPS)
python src/08_final.py                 # ablación, ajuste y clasificación con validación cruzada
python src/08_final.py --evaluar-prueba  # evaluación única de los modelos nuevos (ya hecha)
python src/09_curva_k.py               # hiperparámetros por punto de partida (validación cruzada)
python src/10_agrupamiento.py          # agrupamiento de perfiles de retraso
python src/12_figuras_final.py         # figuras del paper final
python src/11_demo.py                  # genera Entrega final/demo/index.html
```

- La prueba de la entrega final ya se evaluó una vez (`resultados/PRUEBA_FINAL_EVALUADA.txt`, con la huella SHA-256 del código y los datos).
- Paper, diapositivas y demo: carpeta `Entrega final/` (ver su `README.md`).

## Estructura

| Ruta | Contenido |
|---|---|
| `src/01_preprocesamiento.py` | Reglas de limpieza con aserciones y tabla de calidad |
| `src/02_eda.py` | Figuras del análisis exploratorio (solo desarrollo) |
| `src/05_analisis.py` | Curva de error por punto de partida, sensibilidad del umbral e importancia por permutación (validación cruzada); desglose de la prueba con las predicciones guardadas, sin reevaluar |
| `src/06_revision.py` | Análisis de la revisión: comparación entre modelos con el día como unidad (bootstrap, signo, Holm), descomposición del error y R² por subgrupo, sesgo de selección, verificación de los pasos anómalos, línea base por hora y tipo de día (añadida después de la evaluación única) y sensibilidad a la semilla. No reentrena los modelos congelados |
| `src/estilo.py` | Estilo común de las figuras |
| `src/features.py` | Variables causales hasta el punto de control k |
| `src/04_regresion.py` | Líneas base, Ridge, Random Forest, HistGradientBoosting; validación cruzada *forward-chaining*, búsqueda de hiperparámetros, prueba única con IC por día y Wilcoxon |
| `src/07_tiempo_real.py` | Variables en tiempo real en P10: retraso ganado por las vueltas precedentes y velocidad GPS de los buses adelante |
| `src/08_final.py` | Ablación de las variables en tiempo real, ajuste, clasificación de puntualidad y evaluación única de la entrega final |
| `src/09_curva_k.py` | Curva de error por punto de partida con hiperparámetros ajustados para cada k |
| `src/10_agrupamiento.py` | K-means y mezcla gaussiana sobre los perfiles de 18 retrasos |
| `src/11_demo.py`, `src/demo_plantilla.html` | Demo interactiva en un solo HTML |
| `src/12_figuras_final.py` | Figuras del paper final |
| `Entrega final/` | Paper final, diapositivas y demo |
| `resultados/` | Todas las tablas del paper (CSV) y la configuración elegida |
| `figuras/` | Figuras del paper (PDF) |
| `scripts/` | Exploración inicial del E1 (rutas fijas a `/mnt/...`; no se usan en el E2) |
