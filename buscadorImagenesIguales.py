
import os
import cv2
import re
import ReducirRuido
import compara as c
import shutil
import datos
import numpy as np

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
    threshold = 23000000
    if np.abs(b - a).sum() < threshold:
        bandera = True
    else:
        bandera = False
    return bandera

imgAna = 0
imgEli = 0
imgIgu = 0

crearDirectorios()
datos.set_paths('Datos/',rutaGuardar,rutaGuardar)
print 'Cargando imagenes...'
imgs_train, imgs_train_mask = datos.cargar_datos_entrenamiento()
print 'Procesando imagenes...'
imgs_train = datos.preprocess(imgs_train)
imgs_train_mask = datos.preprocess(imgs_train_mask)

lista_marcadas = []
total = len(imgs_train)
for i in range(imgs_train.shape[0]):
    if not(lista_marcadas.__contains__(i)):
        im1 = imgs_train[i][0]
        for j in range(imgs_train.shape[0]):
            if i!=j and not(lista_marcadas.__contains__(j)):
                im2 = imgs_train[j][0]
                if compararImagenes(im1,im2):
                    print 'Coincidencia encontrada.'
                    imgIgu = imgIgu + 1
                    sum_mask1 = imgs_train_mask[i][0].sum()
                    sum_mask2 = imgs_train_mask[j][0].sum()
                    if sum_mask1!=0 and sum_mask2!=0:
                        print 'Las dos son positivas...'
                    elif sum_mask1==0 and sum_mask2==0:
                        print 'Las dos son negativas...'
                    else:
                        lista_marcadas.append(i)
                        lista_marcadas.append(j)
                        print 'Marcadas para eliminar: %d | %d'%(i,j)
                        imgEli = imgEli + 1
    imgAna = imgAna + 1
    print "Analizadas:%d/%d \n Iguales:%d \n Marcadas:%d\n" % (imgAna,total, imgIgu, imgEli)
    print lista_marcadas

datos.crear_datos_sd(lista_marcadas)




"""
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
"""