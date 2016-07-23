import os
import cv2
import ReducirRuido
import shutil
from skimage import img_as_ubyte
import re
import time

rutaImgTrain = os.path.join("Datos","train")
rutaImgTest = os.path.join("Datos","test")
rutaNorm = "DatosNormalizados"
rutaNormTrain = os.path.join(rutaNorm,"train")
rutaNormTest = os.path.join(rutaNorm,"test")

def compara(a, b):
    numeros_a = map(int, re.findall("\d+", a))
    numeros_b = map(int, re.findall("\d+", b))

    if len(numeros_a) == 0:
        return 1

    if numeros_a < numeros_b:
        retorno = -1
    elif numeros_a == numeros_b:
        retorno = 0
    else:
        retorno = 1

    return retorno

def normalizarGuardar(rutaOrigen,rutaDestino):
    for base, dirs, files in os.walk(rutaOrigen):
        files.sort(cmp=compara)
        for name in files:
            print name
            img = cv2.imread(os.path.join(rutaOrigen, name))
            nomArch = name.split('.')[0]
            if not 'mask' in name:
                img = ReducirRuido.denoiseMorfologico(img)

                img = ReducirRuido.denoiseNonLocalMeans(img)

                img = img_as_ubyte(img)

            cv2.imwrite(os.path.join(rutaDestino, nomArch+".png"), img)

if os.path.exists(rutaImgTrain) and os.path.exists(rutaImgTest):
    op = "S"
    if os.path.exists(rutaNorm):
        op = raw_input("Esta seguro que desea eliminar el contenido de la carpeta Entrenamiento(S/n)")
        if op == "S":
           shutil.rmtree(rutaNorm, ignore_errors=True)
    else:
        os.makedirs(rutaNorm)

    if op =="S":
        os.makedirs(rutaNormTest)
        os.makedirs(rutaNormTrain)

        normalizarGuardar(rutaImgTrain,rutaNormTrain)
        normalizarGuardar(rutaImgTest,rutaNormTest)

    print "FIN"
else:
    print "Debe crear una carpeta Datos que contenga dentro las carpetas train y test con sus respectivas" \
          "imagenes"