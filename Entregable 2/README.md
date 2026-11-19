# Entregable 2: implementación y evaluación experimental

Archivos de esta carpeta (lo que se sube a la plataforma):

| Archivo | Contenido |
|---|---|
| `paper-entregable-2.pdf` | Informe (incluye el Entregable 1 actualizado) |
| `Entregable2_codigo.ipynb` | Todo el código del entregable en un solo notebook, ya ejecutado con sus salidas |
| `requirements.txt` | Versiones exactas de las librerías |

## Cómo ejecutar el notebook

1. Instalar las dependencias: `pip install -r requirements.txt`
2. Colocar `Times.csv` junto al notebook. Dentro del repositorio no hace falta: el notebook lo toma de `../Dataset/Times.csv`.
3. Ejecutar todas las celdas en orden (unos 9 min). Las salidas se guardan en `salidas/`.

La última celda comprueba que la evaluación en la prueba coincide con la oficial, hecha con el protocolo congelado en el *commit* `6445b10`.
