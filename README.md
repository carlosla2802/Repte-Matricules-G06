# Repte-Matricules-G06
Repte Matricules - Grup 06 - Mètodes Avançats de Processament de Senyal, Imatge i Vídeo - 2023

## Descripción del proyecto
Este proyecto tiene como objetivo detectar matrículas de coches en imágenes frontales y parcialmente de lado desde una perspectiva de una cámara de parking y reconocer sus caracteres. El código está diseñado para operar en dos modos: modo de inferencia y modo main.

## Modo de inferencia
En este modo, puedes proporcionar una imagen de un coche y observar todos los pasos del proceso de detección y reconocimiento de la matrícula.

#### Instrucciones:
Utiliza la función inference_mode(img_filename, mode="easyocr").

Proporciona la ruta de la imagen del coche como img_filename.

Elige el modo de reconocimiento de caracteres (easyocr o similarity) configurando el parámetro mode.

## Modo main
En este modo, el proceso se ejecuta internamente para todas las imágenes de coches en la base de datos llamada matriculas_db. A partir de ello se rellenarán las carpetas matriculas_recortadas, output_with_easyocr y output_with_similarity con los resultados de cada una de las imágenes.

#### Instrucciones:
Utiliza la función main_matriculas(mode="easyocr").

Elige el modo de reconocimiento de caracteres (easyocr o similarity) configurando el parámetro mode.

## Estructura de archivos y carpetas
**matriculas.py**: Código principal del proyecto.

**matriculas_db**: Base de datos con imágenes de coches.

**alpha**: Base de datos alfanumérica.

**matriculas_recortadas**: Carpeta con imágenes recortadas de las matrículas.

**output_with_easyocr**: Carpeta con resultados del reconocimiento usando EasyOCR.

**output_with_similarity**: Carpeta con resultados del reconocimiento usando el método de similitud.

### Equipo:
Carlos Leta - 1599255@uab.cat,
Abel Espín - 1605961@uab.cat,
Andreu Cuevas - 1570422@uab.cat

