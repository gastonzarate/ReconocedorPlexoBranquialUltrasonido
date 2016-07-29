
import os
import cv2
import re
import ReducirRuido
import compara as c
import shutil

rutaPos = os.path.join("Datos","lista-pos.txt")
rutaNeg = os.path.join("Datos","lista-neg.txt")
rutaCarpeta = os.path.join("DatosNormalizados","train")
rutaGuardar = os.path.join("DatosNormalizados","trainsd")


def cargarDiccionario(ruta):
    archivo = open(ruta,"r")
    dic = {}
    for name in archivo.readlines():
        dic[name.split('.')[0]] = 1
    return dic

def crearDirectorios():
    op = "S"
    # Pregunta si existen los directorios o archivos para eliminarlos
    if os.path.exists(rutaGuardar):
        op = raw_input("Esta seguro que desea eliminar el contenido de la carpeta trainssd(S/n)")
        if op =="S":
            shutil.rmtree(rutaGuardar, ignore_errors=True)
            os.makedirs(rutaGuardar)
    else:
        os.makedirs(rutaGuardar)


def compararImagenes(a,b):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
    img = cv2.subtract(a, b)
    img,thresh = cv2.threshold(img, 15, 255, cv2.THRESH_BINARY)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    print len(contours)
    if(len(contours)==0):
        bandera = True
    else:
        bandera = False
    return bandera

#Carga las imagenes en memoria
imagenes = [], []
dicPos = cargarDiccionario(rutaPos)
dicNeg = cargarDiccionario(rutaNeg)

crearDirectorios()
imgAna = 0
imgEli = 0
imgIgu = 0
for base,dirs,files in os.walk(rutaCarpeta):
    files.sort(cmp=c.compara)
    for name in files:
        if not 'mask' in name:
            nom = name.split('.')
            nueva = cv2.imread(os.path.join(base, name), cv2.IMREAD_GRAYSCALE)

            for i in range(len(imagenes[0])):
                nomComp = imagenes[1][i]
                if compararImagenes(nueva,imagenes[0][i]):
                    imgIgu = imgIgu + 1
                    if (dicPos.get(nom)!=None and dicNeg.get(nomComp!=None) or
                             (dicPos.get(nomComp)!=None and dicNeg.get(nom)!=None)):
                        imagenes[0].pop(i)
                        imagenes[1].pop(i)
                        imgEli = imgEli + 1
                else:
                    imagenes[0].append(nueva)
                    imagenes[1].append(nom)

            if len(imagenes[0])==0:
                imagenes[0].append(nueva)
                imagenes[1].append(nom)
            imgAna = imgAna + 1
            print "Analizadas:%d \n Iguales:%d \n Eliminadas:%d\n" %(imgAna,imgIgu,imgEli)

print "Guardando imagenes\n"
largo = len(imagenes[0])
for i in range(largo):
    print i + " de " + largo
    cv2.imwrite(os.path.join(rutaGuardar,imagenes[1][i]),imagenes[0][i])