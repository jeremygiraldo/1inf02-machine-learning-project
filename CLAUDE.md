# Proyecto final — Aprendizaje de Máquina (1INF02, PUCP, 2026-2)

Contexto para Claude Code. Resume el análisis y las decisiones tomadas hasta el 6 nov 2026.

## Enunciado (resumen)
- Dataset de un sistema de transporte inteligente (STI) de buses de Cusco, entregado por los docentes (Beltrán, Alvarez).
- Plantear una tarea supervisada (clasificación/regresión) o no supervisada (clustering); comparar varios algoritmos con un diseño experimental, métricas y reproducibilidad.
- Herramientas: scikit-learn, PyTorch o TensorFlow. Entrega: paper (formato IEEE recomendado), código (.ipynb/.py, idealmente en GitHub) y demo interactiva (recomendada).
- Fechas: E1 propuesta + antecedentes (6 nov) · E2 implementación y evaluación (20 nov) · presentación final + paper (27 nov, 15 min + 5 de preguntas).

## Datos
### `Times.csv` — 4 707 filas, una por vuelta de un bus
- Columnas: `id, lap, date, time, start_time, end_time, difference_time, vehicle_id_id`.
- `start_time`, `end_time` y `difference_time` son arreglos tipo Postgres `{a,b,...}` con **18 valores** (los 18 puntos de control, P1..P18).
- `start_time` = hora programada, `end_time` = hora real, `difference_time` = `trunc((real − programada) / 60)` en minutos (negativo = adelantado).
- Periodo: 6 ene – 1 feb 2025 (27 días); 45 vehículos; vueltas 1–8 por día.
- 4 plantillas de horario (duración programada 147 / 161 / 168 / 170 min); la de 161 min cubre el 65 % de las vueltas.

### `Monitoring.csv` — 1 630 431 filas, una por posición GPS
- Columnas: `id, date, time, lap, latitude, longitude, speed (km/h), bearing, vehicle_id_id`.
- Periodo: 1 ene – 1 feb 2025; 45 vehículos; intervalo de muestreo mediano 25 s (p90 = 68 s).
- `lap = 0` (2,9 %) = bus fuera de vuelta (terminal o fuera de servicio).

### Unión
- Leer `vehicle_id_id` como **string** (tiene ceros a la izquierda: "05").
- Unir por vehículo + fecha + vuelta; para ubicar cada punto de control: `merge_asof` de la hora real (`end_time`) con el GPS del mismo vehículo, tolerancia de 60 s → empareja el 99,2 % de los pasos; la vuelta coincide en el 99,1 %.
- Es **una sola ruta circular**: sale del extremo este (P1 ≈ −13.5552, −71.8568), va al oeste hasta P10 (≈ −13.5543, −71.9908) y vuelve al este. Coordenadas medianas por punto en `scripts/geo.py` → `cp.csv`.

### Problemas de calidad y reglas de limpieza
- Pasos no registrados: `-` en `difference_time` y `00:00:00` en `end_time`. El 35 % de las vueltas tiene al menos uno; P17 falta en el 27 %. → NaN + indicador; nunca imputar la variable objetivo.
- Pasos asignados a otra vuelta: 156 celdas con |diff| > 60 min (hasta −1 052) en 60 vueltas. **No son cruces de medianoche** (ninguna cerca de ±1440): son vueltas nocturnas con un paso de la mañana siguiente (p. ej. sale 18:52, P18 = 05:45:20). → marcar como NaN. Corregido en el E2; el E1 decía "medianoche". 93 celdas más con 30 < |d| ≤ 60 (zona gris, falta análisis de sensibilidad).
- Tramos con tiempo negativo: 30 vueltas, todas explicadas por la regla anterior (0 tras aplicarla).
- 148 vueltas con horario programado no creciente. → excluir.
- GPS: filtrar `lap ≥ 1` y puntos lejos de la ruta.
- Tras la limpieza: 4 559 vueltas; 4 004 con P18 observado.

## Hallazgos del EDA
- El retraso crece a lo largo de la ruta: mediana 0 / −1 min en P1–P5, de 1 a 3 min desde P14; p75 en P18 = 6 min.
- En P18: 38 % llega > 3 min tarde, 17 % llega ≥ 2 min adelantado.
- Efectos de la hora del día y del día de la semana (los sábados hay más retraso).

## Propuesta actual (entregable 1, ya redactado)
Documento de Claude Docs: https://claude.ai/artifact/MTnxNoyEsm5XcLCbxtvV6A
- **Tarea principal (regresión):** desde el punto k, predecir el retraso en los puntos k+1..18 (experimento base: k = 10 → P18).
- **Complementos:** clasificación adelantado (≤ −2) / puntual (−1..3) / retrasado (≥ 4); K-means/GMM sobre el perfil de 18 retrasos (+ PCA).
- **Modelos:** líneas base (retraso cero, persistencia, media histórica por hora), Ridge/Lasso, SVR, Random Forest, XGBoost/LightGBM, MLP, LSTM sobre la secuencia de tramos. Interpretación con permutación + SHAP.
- **Validación temporal por día:** entrenamiento 6–21 ene, validación 22–26 ene, prueba 27 ene – 1 feb 2025; *forward-chaining* dentro del entrenamiento.
- **Métricas:** MAE (principal), RMSE, R², % con |error| ≤ 2 min; F1 macro / exactitud balanceada; silueta / Davies-Bouldin. Prueba de Wilcoxon + IC por bootstrap.

## Resultados preliminares (`scripts/feas.py`, solo Times, sin GPS ni ajuste de hiperparámetros)
Predicción de P18 desde P1–P10 + hora, día, vuelta, plantilla, vehículo. Train+val 3 082 vueltas, test 922.

| Modelo | MAE | RMSE | R² |
|---|---|---|---|
| Retraso cero | 5,15 | 8,21 | −0,06 |
| Media histórica por hora | 4,28 | 6,02 | 0,43 |
| Persistencia | 3,99 | 5,97 | 0,44 |
| Ridge | 3,58 | 5,00 | 0,61 |
| HistGradientBoosting | 2,85 | 4,22 | 0,72 |
| Random Forest | 2,85 | 4,17 | 0,73 |

- Clasificación (HistGB): F1 macro 0,68.
- K-means (k = 3) sobre 3 069 perfiles completos: silueta 0,60; grupos normal (2 470), ~13 min tarde (367), ~18 min adelantado (232).

## Discusión abierta: diferenciarnos de otros grupos
Todos los grupos tienen el mismo dataset y `difference_time` es la etiqueta obvia, así que es probable que varios propongan predecir el retraso o clasificar la puntualidad. Alternativas consideradas (la etiqueta hay que construirla):
- **Intervalo entre buses / apelotonamiento:** diferencia de hora real entre buses consecutivos en el mismo punto de control. ← candidata fuerte
- **Tiempo detenido en paraderos** desde el GPS (velocidad ≈ 0). ← candidata fuerte
- Perfiles de conducción por vehículo (clustering; "correteo" = adelantarse para ganar pasajeros).
- Reconstruir pasos no registrados usando el GPS.
- Detectar paraderos reales con DBSCAN; detectar vueltas anómalas.
- Identificar el vehículo por su forma de manejar.

Opciones de diferenciación: (1) cambiar el objetivo a apelotonamiento, reutilizando el pipeline; (2) mantener el retraso y diferenciar con variables del GPS + clustering de perfiles de conducción. **Decisión pendiente.**

## Entregable 2 (rama `claudia-E2`, nivel 1 hecho el 19 nov)
- Alcance reducido por indicación del docente: solo regresión P10 → P18 con `Times.csv`; GPS, curva por k, clasificación, clustering y demo pasan a la entrega final.
- Código: `src/01_preprocesamiento.py`, `src/features.py`, `src/04_regresion.py`; resultados en `resultados/`; paper en `paper-entregable-2.tex` (babel con `es-noshorthands` para que compile con TikZ).
- Protocolo: CV forward-chaining por días (11 días iniciales + 5 pliegues de 2 días) en 6–26 ene; prueba 27 ene–1 feb evaluada UNA vez (commit 6445b10). No volver a evaluarla.
- Meta "80 %" del docente interpretada como R² ≥ 0,80. Métrica principal MAE; operativa % |error| ≤ 2 min (del E1).
- Resultado prueba: HistGB MAE 2,77 [2,29; 3,44], R² 0,76 [0,62; 0,83], ≤2 min 49 %; persistencia MAE 4,32. Meta R² no cumplida. El error se concentra el sábado 1 feb (HistGB MAE 4,61).
- "Retraso de la vuelta anterior" evaluado y descartado por la regla de ablación (mejora solo 1 de 3 modelos).
- Nivel 2 (hecho): `src/02_eda.py` (EDA solo desarrollo), `src/05_analisis.py` (todo con CV salvo el desglose de la prueba, que usa predicciones ya guardadas).
  - Curva por k (CV, hiperparámetros de k=10): HistGB R² ≥ 0,80 desde P14 (0,806); en P16–17 Ridge le gana un poco.
  - Umbral 30 vs 60 min: MAE CV 3,20 vs 3,21 en las mismas vueltas → robusto, se mantiene 60.
  - Importancia: hora de salida (seno 1,33 min) > día de la semana (0,74, solo en pliegues con sábado) > d10 (0,32); plantilla ≈ 0.
  - **Cambio de horario**: la plantilla de 170 min (salidas 16–20 h) existe solo hasta el 24 ene; desde el 27 ene la reemplaza una de 168 min que nunca aparece en desarrollo. En la prueba, HistGB MAE 3,37 en esas vueltas vs 2,64 con la de 161.
- Revisión (`src/06_revision.py`, sin reevaluar la prueba):
  - Comparación por día (la unidad es el día, no la vuelta): HistGB gana en 6/6 días a las líneas base y a Ridge; a RF solo en 5/6 (ΔMAE 0,20 [0,02; 0,37], prueba de signo p = 0,22). La Wilcoxon por vuelta era demasiado optimista; queda solo como referencia.
  - Error de prueba: sábado = 11 % de vueltas y 19 % del error; plantilla 168 = 19 % y 24 %; el resto, MAE 2,30.
  - Pasos con |d| > 60: 122/141 negativos tienen hora de reloj 4–6 h en vueltas que salen 16–20 h; **no** coinciden con pasos de otras vueltas (13 % a ≤ 2 min vs 33 % al azar) → la regla se renombró "paso con hora incompatible"; "de otra vuelta" no está verificado.
  - Vueltas sin P18 (11 % del desarrollo): salen más tarde y algo más en sábado, pero d14/d16 son similares.
  - R² ≥ 0,80 desde P10 "no se demuestra" (el IC de la prueba lo incluye; ningún pliegue de CV llega: máximo 0,775).
- Pendiente para la final: GPS, clasificación, clustering, demo, hiperparámetros por k.

## Scripts (`scripts/`)
Las rutas a los CSV están fijas (`/mnt/user-data/...`); hay que cambiarlas a las rutas locales.
- `eda.py`, `eda2.py`: exploración inicial de ambos archivos.
- `clean.py`: limpieza + primera prueba de modelos.
- `geo.py`: une puntos de control con el GPS y obtiene sus coordenadas (`cp.csv`, `sample.csv`).
- `more.py`: cuantiles por punto, prueba de K-means y mapa de la ruta (`ruta.png`).
- `feas.py`: prueba de factibilidad final con la partición propuesta.
