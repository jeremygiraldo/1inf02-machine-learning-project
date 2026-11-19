# Entregable 2: implementación y evaluación experimental

Archivos de esta carpeta (lo que se sube a la plataforma):

| Archivo | Contenido |
|---|---|
| `paper-entregable-2.pdf` | Informe (incluye el Entregable 1 actualizado) |
| `paper-entregable-2.tex` | Fuente LaTeX del informe |
| `figuras/` | Figuras que usa el `.tex` (necesarias para compilarlo) |
| `Entregable2_codigo.ipynb` | Todo el código del entregable en un solo notebook, ya ejecutado con sus salidas |
| `requirements.txt` | Versiones exactas de las librerías |

## Cómo ejecutar el notebook

1. Instalar las dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Colocar `Times.csv` junto al notebook. Dentro del repositorio no hace falta: el notebook lo toma de `../Dataset/Times.csv`.
3. Ejecutar todas las celdas en orden (unos 9 min). Las salidas se guardan en `salidas/`.

La última celda comprueba que la evaluación en la prueba coincide con la oficial, hecha con el protocolo congelado en el *commit* `6445b10`.

## Cómo convertir el `.tex` a PDF

El `.tex` necesita la carpeta `figuras/` al lado. Usa la clase `IEEEtran` y el paquete `babel` en español, y se compila con **pdfLaTeX**.

### Opción A: Overleaf (sin instalar nada)

1. Comprimir en un `.zip` el archivo `paper-entregable-2.tex` y la carpeta `figuras/`.
2. Entrar a <https://www.overleaf.com> y crear una cuenta (o iniciar sesión).
3. Elegir **New Project → Upload Project** y subir el `.zip`.
4. Abrir **Menu** (arriba a la izquierda) y revisar dos opciones:
   - **Compiler:** `pdfLaTeX`
   - **Main document:** `paper-entregable-2.tex`
5. Pulsar **Recompile**. El PDF aparece a la derecha.
6. Descargarlo con el ícono de descarga, junto a **Recompile**.

### Opción B: en tu computadora

1. Instalar una distribución de LaTeX:
   - **Windows:** MiKTeX (<https://miktex.org/download>). Durante la instalación, elegir que instale los paquetes que falten automáticamente.
   - **macOS:** MacTeX (<https://tug.org/mactex/>).
   - **Linux (Debian/Ubuntu):**
     ```bash
     sudo apt install texlive-latex-recommended texlive-latex-extra texlive-publishers texlive-lang-spanish texlive-fonts-recommended texlive-pictures latexmk
     ```
2. Abrir una terminal en esta carpeta (`Entregable 2`).
3. Compilar:
   ```bash
   latexmk -pdf paper-entregable-2.tex
   ```
   `latexmk` compila las veces necesarias para que las referencias y la numeración de tablas y figuras queden bien.

   Si no tienes `latexmk`, ejecuta `pdflatex` **dos veces**:
   ```bash
   pdflatex paper-entregable-2.tex
   ```
   ```bash
   pdflatex paper-entregable-2.tex
   ```
4. El resultado es `paper-entregable-2.pdf`, en la misma carpeta.
5. Opcional: borrar los archivos auxiliares (`.aux`, `.log`, `.out`, …):
   ```bash
   latexmk -c
   ```

### Problemas frecuentes

- **Un `??` en lugar de un número de tabla o figura:** falta una segunda compilación. Compila otra vez, o usa `latexmk`.
- **`File 'IEEEtran.cls' not found` o un paquete faltante:** instala el paquete (con MiKTeX se instala solo; en Linux, `texlive-publishers` trae IEEEtran).
- **Una figura sale como recuadro con el texto "Insertar…":** la carpeta `figuras/` no está junto al `.tex`.
