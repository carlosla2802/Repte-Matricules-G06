import numpy as np 
import cv2
import imutils
import easyocr
import os
import matplotlib.pyplot as plt
import logging

logging.getLogger('easyocr.easyocr').setLevel(logging.ERROR)


"""
Detector de matriculas:

    - Pasas como input una foto de un coche donde la matricula sea visible y te devuelve los pixeles de la matricula
"""
def detect_license_plate(img_filename, show_plots=False):


    # Cargar la imagen en escala de grises y en color
    gray_image = cv2.imread(img_filename, 0)
    original_gray_image = gray_image.copy()
    color_image = cv2.imread(img_filename)
    original_color_image = color_image.copy()

    # Redimensionar las imágenes
    resized_gray = resize_image(gray_image)
    resized_color = resize_image(color_image)

    # Mostrar la imagen original si se solicita
    if show_plots:
        cv2.imshow('Original Image', original_color_image)
        cv2.waitKey(0)

    # Aplicar transformaciones para preparar la imagen para la detección de contornos
    transformed_image = apply_transformations(resized_gray)

    if show_plots:
        cv2.imshow('Transformed Image', transformed_image)
        cv2.waitKey(0)

    # Obtener contornos de la imagen transformada
    contours_image = get_contours(transformed_image)

    if show_plots:
        cv2.imshow('Image with Contours', contours_image)
        cv2.waitKey(0)

    # Localizar la matrícula en la imagen
    license_plate_location = find_license_plate(contours_image)
    
    # Si no la encuentra no seguimos con el proceso
    if license_plate_location is None:
        return
    
    
    itemp = color_image.copy()

    if show_plots==True:
        cv2.drawContours(itemp, [license_plate_location], -1, (124, 252, 0), 2)
        cv2.imshow('Matrícula Localizada', itemp)
        cv2.waitKey(0)

    loc = license_plate_location.copy()

    x = license_plate_location[0][0]
    y = license_plate_location[0][1]
    w = license_plate_location[2][0] - x
    h = license_plate_location[3][1] - y
    
    cv2.drawContours(itemp, [license_plate_location], -1, (124, 252, 0), 2)

    mask = np.zeros(original_gray_image.shape, np.uint8)
    y, x = original_gray_image.shape

    for i in range(4):
        license_plate_location[i][0] = license_plate_location[i][0] / (720/x)
        license_plate_location[i][1] = license_plate_location[i][1] / (720/x)

    new_image = cv2.drawContours(mask, [license_plate_location], 0, 255, -1)
    new_image = cv2.bitwise_and(original_color_image, original_color_image, mask=mask)

    # Aplicamos la homografía
    warped = homography(new_image, license_plate_location)

    if show_plots==True:
        cv2.imshow('Imagen corregida', warped)
        cv2.waitKey(0)

    return warped

def resize_image(img):

    shape = img.shape
    y, x = shape[0], shape[1]

    r = 720 / x
    dim = (720, int(y * r))
    img = cv2.resize(img, dim)

    return img



def find_license_plate(image):
    keypoints = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    num_contours = 10 if len(contours) > 10 else len(contours)
    contours = sorted(contours, key=cv2.contourArea,
                      reverse=True)[:num_contours]

    location = None
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        approx = cv2.boxPoints(rect)
        approx = np.int0(approx)
        if len(approx) == 4:
            points = np.squeeze(approx)

            X, Y = points[:, 0], points[:, 1]

            left_y, right_y = np.min(Y), np.max(Y)
            top_x, bottom_x = np.min(X), np.max(X)

            x, y = right_y - left_y, bottom_x - top_x
            ratio = y / x

            if 8 > ratio > 1.8:
                location = approx
                break

    return location

def get_contours(imgTransformed):
    # Filtro usando contour area y quitando ruido

    cnts = cv2.findContours(
        imgTransformed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 1000:
            cv2.drawContours(imgTransformed, [c], -1, (0, 0, 0), -1)

    return imgTransformed


def apply_transformations(image):

    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blackHat = cv2.morphologyEx(
        blurred, cv2.MORPH_BLACKHAT, kernel, iterations=3)

    treshold = cv2.threshold(blackHat, blackHat.max() //
                             2, 255, cv2.THRESH_BINARY)[1]

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 1))
    dilate = cv2.dilate(treshold, horizontal_kernel, iterations=7)

    opening = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, kernel, iterations=1)

    gaussianBlur = cv2.GaussianBlur(opening, (3, 3), 0)

    treshold2 = cv2.threshold(
        gaussianBlur, gaussianBlur.max()//2, 255, cv2.THRESH_BINARY)[1]

    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closing = cv2.morphologyEx(
        treshold2, cv2.MORPH_CLOSE, square_kernel, iterations=5)

    return closing

def order_points(pts):
    """
    Recibe los 4 puntos de los vertices de la matricula y los ordena desde la esquina superior izquierda, 
    pasando por la superior derecha y luego por la inferior derecha, hasta llegar a la inferior izquierda.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def homography(gray_img, location):
    """
    Transformación de perspectiva (homografía) en la imagen para aislar 
    y corregir la orientación de lamatrícula. 
    
    La transformación asegura que la matrícula aparezca en la vista frontal, 
    haciendo más fácil su posterior análisis o reconocimiento.
    """
    # Corrige rotación
    src = order_points(np.squeeze(location).astype(np.float32))
    (tl, tr, br, bl) = src
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray_img, M, (maxWidth, maxHeight))

    return warped
                

#----------------------------------------------------------------------------------------------------

"""
Pasas como input la imagen de la matricula (una vez se a recortado) y utiliza easyocr para devolverte el str con la matricula
"""

def img_to_str_easyocr(img_filename, show_plots=False):
    # Convertir la imagen a escala de grises y binarizar
    img = cv2.imread(img_filename, cv2.IMREAD_COLOR)

    #Ocultar parte azul matricula
    thresh = hide_blue(img, show_plots=show_plots)

    if show_plots==True:
        cv2.imshow('Imagen trás ocultar azules', thresh)
        cv2.waitKey(0)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    if show_plots==True:
        cv2.imshow('Imagen final (input easyocr)', thresh)
        cv2.waitKey(0)

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(thresh, detail=0, allowlist='0123456789BCDFGHJKLMNPRSTVWXYZ', paragraph=True)

    if len(result) == 0:
        result = ''

    elif len(result) >= 1:
        result = ''.join(result).replace(" ", "")

    # Reemplazar caracteres similares cuando no corresponden a su posición
    number_text_similarity = (("L", "4"), ("S", "5"),
                              ("Z", "2"), ("B", "8"), ("G", "6"))
    
    numbers, letters = result[:4], result[4:]

    for pair in number_text_similarity:
        numbers = numbers.replace(pair[0], pair[1])
        letters = letters.replace(pair[1], pair[0])

    result_modif = numbers + letters

    if show_plots==True:
        if result != result_modif:
            print(f'La matricula detectada por easyocr es: {result}, se a actualizado a {result_modif} utilizando un number-text similarity dict')
    
    return result_modif


"""
Pasas como input la imagen de la matricula (una vez se a recortado) y realiza un calculo de similitud caracter a caracter con una base de datos propia de caracteres alfanumericos (METODO PROPIO PARA COMPARAR CON EASYOCR)
"""

def img_to_str_similarity(img_filename, show_plots=False):
    # Preprocesado imagen matrícula
    license_plate_image = img_filename
    img, contours = preprocess_image(license_plate_image, show_plots=show_plots)

    # Cargar imágenes alpha
    alpha_folder = "Alpha"
    alpha_images = load_alpha_images(alpha_folder)

    # Cargar una imagen alpha para obtener sus dimensiones
    sample_alpha_path = os.path.join(alpha_folder, 'A.bmp')  # Reemplaza 'A.bmp' con un archivo que sepas que existe en la carpeta
    sample_alpha_img = cv2.imread(sample_alpha_path, 0)
    alpha_height, alpha_width = sample_alpha_img.shape

    license_plate_str = ""

    if show_plots==True:
        # Loop through each segment in the license plate
        fig, axes = plt.subplots(1, len(contours), figsize=(20, 20))

    for i, contour in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
        segment = img[y:y+h, x:x+w]

        # Definir el grosor del borde
        top, bottom, left, right = [1]*4
        # Añadir un borde blanco alrededor del segmento
        segment_with_border = cv2.copyMakeBorder(segment, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        #Binarizamos los segmentos
        gray_segment = cv2.cvtColor(segment_with_border, cv2.COLOR_BGR2GRAY)
        _, binarized_segment = cv2.threshold(gray_segment, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Aplicar un filtro Gaussiano para suavizar la imagen binaria
        smoothed_segment = cv2.GaussianBlur(binarized_segment, (5, 5), 0)

        # Redimensionar el segmento para que coincida con las dimensiones de las imágenes alpha
        aspect_ratio = w / h
        new_width = int(alpha_height * aspect_ratio)
        resized_segment = cv2.resize(smoothed_segment, (new_width, alpha_height))
        

        max_similarity = -1
        best_match = None
        
        if show_plots==True:
            # Mostrar el segmento
            axes[i].imshow(resized_segment, cmap='gray')
            axes[i].axis('off')
   
        # Calcular similarity con cada imagen alpha
        for char, char_img in alpha_images.items():
            similarity = calculate_similarity(resized_segment, char_img)
            if similarity > max_similarity:
                max_similarity = similarity
                best_match = char
        

        license_plate_str += best_match
    
    if show_plots==True:
        plt.show()


    return license_plate_str
        
#UTILS DE IMG_TO_STR

def preprocess_image(image_path, show_plots=False):
    # Cargar la imagen en color
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    #Ocultar parte azul matricula
    thresh = hide_blue(img,show_plots=show_plots)

    if show_plots==True:
        cv2.imshow('Imagen gray tras ocultar azules', thresh)
        cv2.waitKey(0)

    # Aplicar operaciones morfológicas
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations = 1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)

    if show_plots==True:
        cv2.imshow('Imagen trás aplicar operaciones morfológicas', thresh)
        cv2.waitKey(0)
    
    contours, _ = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar por área y coger los 7 más grandes
    sorted_by_area_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:7]
    
    filtered_contours = []
    for contour in sorted_by_area_contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w)/h
        
        # Asumiendo que la relación de aspecto de un caracter válido esté entre 0.2 y 1.0
        if 0.2 <= aspect_ratio <= 0.8:
            filtered_contours.append(contour)


    sorted_by_position_contours = sorted(filtered_contours, key=lambda ctr: cv2.boundingRect(ctr)[0])

    return img, sorted_by_position_contours

def hide_blue(img, show_plots=False):

    # Convertir a Espacio de Color HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Reemplazar Píxeles Azules con Blanco:
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(img_hsv, lower_blue, upper_blue)
    img[mask != 0] = [255, 255, 255]

    if show_plots==True:
            cv2.imshow('Imagen tras ocultar azules', img)
            cv2.waitKey(0)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh

def load_alpha_images(alpha_folder):
    alpha_images = {}
    for filename in os.listdir(alpha_folder):
        char = filename.split('.')[0]  # Asumiendo nombres de archivo como "A.bmp", "B.bmp", etc.
        img_path = os.path.join(alpha_folder, filename)
        img = cv2.imread(img_path, 0)
        
        if img is None:
            #print(f"Error al cargar la imagen: {img_path}")
            continue
        
        alpha_images[char] = img

    return alpha_images

def calculate_similarity(segment, char_img):
    if char_img is None:
        #print("Imagen de caracter vacía. Saltando...")
        return 0
    
    # Resize char_img al mismo size del segmento
    char_img = cv2.resize(char_img, (segment.shape[1], segment.shape[0]))
    
    # Calcular la correlación cruzada
    correlation = cv2.matchTemplate(segment, char_img, cv2.TM_CCOEFF_NORMED)    
    return np.max(correlation)


"""
Todas las fotos de matriculas guardadas en la bd se procesaran para obtener la matricula en formato str
"""

def main_matriculas(mode="easyocr"):
    # Iterar sobre los nombres de archivos en la carpeta
    carpeta_input = "matriculas_db"
    carpeta_output = "matriculas_recortadas"
    for filename in os.listdir(carpeta_input):
        try:
            imagen_recortada = detect_license_plate(os.path.join(carpeta_input, filename), show_plots=False)
            
            # Genera el nombre de archivo de salida con "_recortada" agregado al nombre original
            nombre_archivo_salida = os.path.splitext(filename)[0] + "_recortada.jpg"
            
            # Ruta completa del archivo de salida
            ruta_archivo_salida = os.path.join(carpeta_output, nombre_archivo_salida)
            
            # Guarda la imagen recortada en la carpeta de salida
            cv2.imwrite(ruta_archivo_salida, imagen_recortada)
            
            #print(f"Imagen recortada guardada en: {ruta_archivo_salida}")
        
        except:
            continue


    #nombres_de_archivos_input = os.listdir(carpeta_input) # Obtener la lista de nombres de archivos en la carpeta de entrada
    #cantidad_de_archivos_input = len(nombres_de_archivos_input) # Contar la cantidad de archivos
    #print(cantidad_de_archivos_input)
    #nombres_de_archivos_output = os.listdir(carpeta_output) # Obtener la lista de nombres de archivos en la carpeta de salida
    #cantidad_de_archivos_output = len(nombres_de_archivos_output) # Contar la cantidad de archivos
    #print(cantidad_de_archivos_output)

    #--------------------------------------------------------------------------------------------------------------------------------
    #img to str

    if mode == "easyocr":
        # Iterar sobre los nombres de archivos en la carpeta
        carpeta_input = "matriculas_recortadas"
        carpeta_output = "output_with_easyocr"

        for filename in os.listdir(carpeta_input):
            #leemos imagen de la matricula recortada
            license_plate_image = os.path.join(carpeta_input, filename)

            #aplicamos easyocr a la imagen
            license_plate_str = img_to_str_easyocr(license_plate_image, show_plots=False)

            #Guardar matricula str en txt
            output_filename = os.path.splitext(filename)[0]
            output_filename = output_filename[:-len("_recortada")] if output_filename.endswith("_recortada") else output_filename
            output_filename += "_easyocr.txt"
            ruta_archivo = os.path.join(carpeta_output, output_filename)
            # Abrir el archivo en modo de escritura ('w' significa write)
            with open(ruta_archivo, 'w') as archivo:
                # Escribir el string en el archivo
                archivo.write(license_plate_str)

    else:
        # Iterar sobre los nombres de archivos en la carpeta
        carpeta_input = "matriculas_recortadas"
        carpeta_output = "output_with_similarity"
        for filename in os.listdir(carpeta_input):
  
            # Preprocesado imagen matrícula
            license_plate_image = os.path.join(carpeta_input, filename)

            license_plate_str = img_to_str_similarity(license_plate_image, show_plots=False)
            
            #Guardar matricula str en txt
            output_filename = os.path.splitext(filename)[0]
            output_filename = output_filename[:-len("_recortada")] if output_filename.endswith("_recortada") else output_filename
            output_filename += "_similarity.txt"
            ruta_archivo = os.path.join(carpeta_output, output_filename)
            # Abrir el archivo en modo de escritura ('w' significa write)
            with open(ruta_archivo, 'w') as archivo:
                # Escribir el string en el archivo
                archivo.write(license_plate_str)

"""
Modo interactivo, le pasas la ruta de una imagen y te iran apareciendo plots de manera
visual para ver todo el proceso de detección de matriculas (NO SE GUARDA EL RESULTADO)
"""

def inference_mode(img_filename, mode="easyocr"):
    directorio = "tmp"
    imagen_path = os.path.join(directorio, "img.jpg")

    #Directorio temporal para guardar imagenes (luego las borraremos)
    if not os.path.exists(directorio):
        os.mkdir(directorio)

    # Recortar matricula
    imagen_recortada = detect_license_plate(img_filename, show_plots=True)

    # Guardamos la imagen recortada en el archivo tmp temporal
    cv2.imwrite(imagen_path, imagen_recortada)

    if mode == "easyocr":
        # Aplicar EasyOCR a la imagen
        license_plate_str = img_to_str_easyocr(imagen_path, show_plots=True)
    else:
        license_plate_str = img_to_str_similarity(imagen_path, show_plots=True)

    print(license_plate_str)

    # Eliminar todos los archivos del directorio temporal
    for filename in os.listdir(directorio):
        file_path = os.path.join(directorio, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    # Eliminar el directorio temporal
    os.rmdir(directorio)








# ----------- EJECUCIÓN MAIN o INFERENCE -----------
# ----- mode = "easyocr" o mode = "similarity" -----

# INFERENCE MODE
#inference_mode("matriculas_db/im3.jpg", mode="easyocr")

# MAIN MODE
main_matriculas(mode="easyocr")



# ---------------------------------------------------------------
# Contador de archivos
#directorio = "output_with_easyocr"
directorio = "output_with_similarity"

contador_txt = 0
contador_7_digitos = 0


for archivo in os.listdir(directorio):
    if archivo.endswith(".txt"):
        contador_txt += 1
        
        with open(os.path.join(directorio, archivo), 'r') as f:
            texto = f.read().strip()  # Leer el contenido del archivo .txt y eliminar espacios en blanco
            
        # Contar el número de caracteres en el archivo .txt
        num_digitos = len(texto)
        
        if num_digitos == 7:
            contador_7_digitos += 1

print(f"Número de archivos .txt en {directorio}: {contador_txt}")
print(f"Número de archivos .txt en {directorio} y con los 7 digitos: {contador_7_digitos}")
