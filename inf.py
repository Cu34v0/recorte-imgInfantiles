import os
import math
import cv2 as cv
import numpy as np

# Convertir imagen a RGB
def toRGB(img):
    '''Convierte imagen de BGR a RGB'''
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Detección de ojos para encontrar ángulo
def encontrar_angulo(img):
    '''Encuentra ojos en la imagen'''
    cascade = cv.CascadeClassifier('haarcascade_eye.xml')
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ojos = cascade.detectMultiScale(gris, 2.1, 13)
    punto1 = (ojos[0][0] + ojos[0][2] // 2, ojos[0][1] + ojos[0][3] // 2)
    punto2 = (ojos[1][0] + ojos[1][2] // 2, ojos[1][1] + ojos[1][3] // 2)
    angulo = math.degrees(math.atan((punto2[1] - punto1[1]) / (punto2[0] - punto1[0])))
    return angulo

# Rota la imagen de acuerdo al ángulo
def rotar(imagen, angulo):
    '''Rota la imagen con respecto a su centro'''
    alto, ancho, nada = imagen.shape
    matriz = cv.getRotationMatrix2D((ancho/2, alto/2), angulo, 1)
    img = cv.warpAffine(imagen, matriz, (ancho, alto))
    return img

# Encuentra el rostro y recorta la imagen a tamaño infantil a 240dpi 
def cortar(img):
    '''Encuentra el rostro'''
    cascade = cv.CascadeClassifier('lbpcascade_frontalface.xml')
    gris = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gris, 1.1, 5)
    x, y, w, h = faces[0][0], faces[0][1], faces[0][2], faces[0][3]
    centro = (x + w // 2, y + h // 2)
    esc = 360 / h
    esqsi = (centro[0] - int(300 / esc), centro[1] - int(400 / esc))
    esqid = (esqsi[0] + int(600 / esc), esqsi[1] + int(720 / esc))
    recorte = img[esqsi[1]:esqid[1], esqsi[0]:esqid[0]]
    redim = cv.resize(recorte, (600, 720), interpolation=cv.INTER_AREA)
    return redim

# Hace un pequeño ajuste de niveles de luz en la imagen
def niveles(img):
    '''Ajuste de niveles'''
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=0.5, tileGridSize=(4, 4))
    l = clahe.apply(l)
    lab = cv.merge((l, a, b))
    return cv.cvtColor(lab, cv.COLOR_LAB2BGR)

def procesar_carpeta(carpeta_entrada, carpeta_salida):
    # Asegúrate de que las carpetas de salida existan
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)

    # Itera sobre cada archivo en la carpeta de entrada
    for archivo in os.listdir(carpeta_entrada):
        if archivo.endswith(('.jpg', '.jpeg', '.png')):
            # Cargar la imagen original
            ruta_original = os.path.join(carpeta_entrada, archivo)
            original = cv.imread(ruta_original)

            # Encuentra el ángulo de rotación
            angulo = encontrar_angulo(original)

            # Rotar la imagen
            enderezada = rotar(original, angulo)

            # Encuentra el rostro y recorta
            recortada = cortar(enderezada)

            # Ajuste de niveles
            ajustada = niveles(recortada)

            # Guarda la imagen resultante en la carpeta de salida (sobrescribe la original)
            ruta_salida = os.path.join(carpeta_salida, archivo)
            cv.imwrite(ruta_salida, ajustada)

# Especifica las carpetas de entrada y salida
carpeta_entrada = 'Img/'
carpeta_salida = 'ImgSalida/'

# Procesa la carpeta
procesar_carpeta(carpeta_entrada, carpeta_salida)