import os
import cv2
import numpy as np

rutaOrigen = os.path.join("DatosNormalizados","train")
clasificador = os.path.join("Clasificadores",'cascade4.xml')
rutaPos = os.path.join("Entrenamiento","pos.info")
rutaNeg = os.path.join("Entrenamiento","neg.info")


listaPos = [],[]
listaNeg = []

#Cargo la lista de negativos
archivo = open(rutaNeg,"r")
for name in archivo.readlines():
    nombre = name.split("\\")[2].replace(" ", "").replace("\n", "")
    listaNeg.append(nombre)

#Cargo la lista de positivos con su posicion
archivo = open(rutaPos,"r")
for linea in archivo.readlines():
    aux = linea.split()
    nombre = aux[0].split("\\")[2]
    listaPos[0].append(nombre)
    listaPos[1].append((aux[2],aux[3],aux[4],aux[5]))

#Carga el clasificador
cascade = cv2.CascadeClassifier(clasificador)
rp = 0
rn = 0
fp = 0
fn = 0
for i in range(len(listaPos[0])):
    #Leo una imagen positiva y negativa
    imgPos = cv2.imread(os.path.join(rutaOrigen,listaPos[0][i]),cv2.IMREAD_GRAYSCALE)
    imgNeg = cv2.imread(os.path.join(rutaOrigen,listaNeg[i]),cv2.IMREAD_GRAYSCALE)
    #La recorto en el area positiva a ambas
    x = int(listaPos[1][i][0]) +10
    y = int(listaPos[1][i][1]) +10
    h = int(listaPos[1][i][2]) +10
    w = int(listaPos[1][i][3]) +10
    yf = y + h
    xf = x + w
    imgPosR = imgPos[y:yf , x:xf]
    imgNegR = imgNeg[y:yf , x:xf]

    #Intento detectar en la imagen positiva y en la negativa
    paPos = cascade.detectMultiScale(imgPosR)
    paNeg = cascade.detectMultiScale(imgNegR)

    if len(paPos) > 0:
        rp = rp + 1
    else:
        fp = fp + 1

    if len(paNeg) == 0:
        rn = rn + 1
    else:
        fn = fn + 1

    print "RealesPositivos:%d - Falsos Positivos:%d - Reales Negativos:%d - FalsosNegativos:%d" %(rp,fp,rn,fn)

preTotal = len(listaPos[0])*2
exactitud = (rp+rn)/ preTotal
presicion = rp / (rp + fp)
sencibilidad = rp / (rp + fn)
especificidad = rn / (rn + fp)

print "\n\n"

print "Exactitud:%d (proximidad entre el resultado y la clasificacion exacta)" %exactitud
print "Presicion:%d (calidad de la respuesta positiva del clasificador)" %presicion
print "Sencibilidad:%d (eficiencia en la clasificacion de todos los elementos que son de la clase)" %sencibilidad
print "Especificidad:%d (eficiencia en la clasificacion de los elementos que no son de la clase)" %especificidad
print "TruePositiveRate:%d (igual a la sencibilidad)" %sencibilidad
print "FalsePositiveRate:%d (1-especificidad)" %(1-especificidad)

