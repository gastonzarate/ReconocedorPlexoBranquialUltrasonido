import os
import cv2
import ReducirRuido
import shutil
from skimage import img_as_ubyte
import time

rutaImgTrain = os.path.join("Datos","Train")
rutaImgTest = os.path.join("Datos","Test")
rutaNorm = "DatosNormalizados"
rutaNormTrain = os.path.join(rutaNorm,"Train")
rutaNormTest = os.path.join(rutaNorm,"Test")

def normalizarGuardar(rutaOrigen,rutaDestino):
    for base, dirs, files in os.walk(rutaOrigen):
        for name in files:
            print name
            img = cv2.imread(os.path.join(rutaOrigen, name))
            nomArch = name.split('.')[0]
            if not 'mask' in name:
                #cv2.imshow("imagen",img)
                #cv2.waitKey(0)

                img = ReducirRuido.denoiseMorfologico(img)
                #cv2.imshow("imagen", img)
                #cv2.waitKey(0)

                img = ReducirRuido.denoiseNonLocalMeans(img)
                #cv2.imshow("imagen", img)
                #cv2.waitKey(0)

                img = img_as_ubyte(img)
                #cv2.imshow("imagen", img)
                #cv2.waitKey(0)

            cv2.imwrite(os.path.join(rutaDestino, nomArch+".png"), img)

if os.path.exists(rutaImgTrain) and os.path.exists(rutaImgTest):
    if os.path.exists(rutaNorm):
        shutil.rmtree(rutaNorm, ignore_errors=True)
    else:
        os.makedirs(rutaNorm)

    os.makedirs(rutaNormTest)
    os.makedirs(rutaNormTrain)

    normalizarGuardar(rutaImgTrain,rutaNormTrain)
    normalizarGuardar(rutaImgTest,rutaNormTest)
    print "FIN"
else:
    print "Debe crear una carpeta Datos que contenga dentro las carpetas train y test con sus respectivas" \
          "imagenes"