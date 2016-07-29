import os
import cv2

rutaOrigen = os.path.join("DatosNormalizados","train")
clasificador = os.path.join("Clasificadores",'cascade5_lbp.xml')
rutaPos = os.path.join("Datos","lista-pos.txt")
rutaNeg = os.path.join("Datos","lista-neg.txt")


def cargarDiccionario(ruta):
    archivo = open(ruta,"r")
    dic = {}
    for name in archivo.readlines():
        dic[name.split('.')[0]] = 1
    return dic


dicPos = cargarDiccionario(rutaPos)
dicNeg = cargarDiccionario(rutaNeg)

#Carga el clasificador
cascade = cv2.CascadeClassifier(clasificador)
for base, dirs, files in os.walk(rutaOrigen):
    for name in files:
        img = cv2.imread(os.path.join(rutaOrigen, name))
        pa = cascade.detectMultiScale(
            img,
            scaleFactor=1.01,
            minNeighbors=5,
            minSize=(5, 5),
            maxSize=(300, 300),
            flags=0)
        # Guarda en el archivo su numero

