# Entrega final: presentación, paper y demo

Archivos de esta carpeta (lo que se sube a la plataforma):

| Archivo | Contenido |
|---|---|
| `paper-final.pdf` | Paper final (incluye el E1 y el E2 actualizados) |
| `paper-final.tex` | Fuente LaTeX del paper (necesita `figuras/` al lado) |
| `diapositivas.pdf` | Diapositivas de la exposición (15 min) |
| `demo/index.html` | Demo interactiva: un solo archivo, se abre con doble clic y funciona sin internet |

El código está en `src/` del repositorio (ver el `README.md` de la raíz).

## Cómo reproducir los resultados de la entrega final

Requiere `Dataset/Times.csv` (versionado) y `Dataset/Monitoring.csv` (GPS, no versionado por su tamaño: colocarlo en `Dataset/`).

```bash
pip install -r requirements.txt
python src/01_preprocesamiento.py      # limpieza
python src/07_tiempo_real.py           # variables en tiempo real (otras vueltas + GPS)
python src/08_final.py                 # ablación, ajuste y clasificación (solo validación cruzada)
python src/09_curva_k.py               # hiperparámetros por punto de partida (validación cruzada)
python src/10_agrupamiento.py          # agrupamiento de perfiles de retraso
python src/12_figuras_final.py         # figuras
python src/11_demo.py                  # genera demo/index.html
```

La prueba de la entrega final ya se evaluó una vez (`resultados/PRUEBA_FINAL_EVALUADA.txt`, con la huella SHA-256 del código y los datos). `08_final.py --evaluar-prueba` se niega a repetirla salvo con `--forzar`.

## Cómo compilar el `.tex`

Igual que en el Entregable 2 (ver `Entregable 2/README.md`): Overleaf con pdfLaTeX, o `latexmk -pdf paper-final.tex`. También funciona `tectonic paper-final.tex`.
