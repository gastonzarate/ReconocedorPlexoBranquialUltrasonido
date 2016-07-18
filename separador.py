import os
import cv2
import re
import ReducirRuido
import numpy as np
from skimage import io

lista_img = [],[]
rutaPos = "pos"
rutaNeg = "neg"
archPos = "pos.info"
archNeg = "neg.info"

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

def listarImagenes():
    for base, dirs, files in os.walk('train'):
        files.sort(cmp=compara)
        for name in files:
            if 'mask' in name:
                lista_img[1].append(name)
            else:
                lista_img[0].append(name)
    return lista_img

def crearDirectorios():
    # Pregunta si existen los directorios o archivos para eliminarlos
    if os.path.exists(rutaPos):
        os.rmdir(rutaPos)
    if os.path.exists(rutaNeg):
        os.rmdir(rutaNeg)
    if os.path.exists(archPos):
        os.remove(archPos)
    if os.path.exists(archNeg):
        os.remove(archNeg)

    # Crea las carpetas
    os.makedirs(rutaPos)
    os.makedirs(rutaNeg)
    archivoP = open(archPos, 'w')
    archivoP.close()
    archivoN = open(archNeg, 'w')
    archivoN.close()

def separarPosNeg():
    # Crea los archivos
    crearDirectorios()

    # Abre los archivos para escribir en ellos
    archivoP = open(archPos, 'a')
    archivoN = open(archNeg, 'a')

    lista = listarImagenes()
    iNeg = 1
    iPos = 1

    for mascara in lista[1]:
        #mask = cv2.imread('train/'+mascara,0)
        mask = io.imread('train/'+mascara,as_grey=True)
        index = lista[1].index(mascara)
        item = lista[0].__getitem__(index)
        img = cv2.imread('train/'+item,cv2.IMREAD_GRAYSCALE)
        imagen = ReducirRuido.denoiseMorfologico(img)
        imagen = ReducirRuido.denoiseNonLocalMeans(imagen)
        nomArch = item.split('.')[0]

        if mask.any()!=0:
            mask, thresh = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
            mask, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # Al encontrarse contorno se busca el de mayor area
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            # Se obtiene la recta correspondiente al contorno
            x, y, w, h = cv2.boundingRect(cnt)
            # Se almacena como positiva
            io.imsave(os.path.join(rutaPos, nomArch + ".png"), imagen)
            archivoP.write(os.path.join(rutaPos, nomArch + ".png") + ' 1 %d %d %d %d \n' % (x, y, h, w))

        else:
            io.imsave(os.path.join(rutaNeg, nomArch + ".png"), imagen)
            archivoN.write(os.path.join(rutaNeg, nomArch + ".png") + '\n')

    # Se cierran los archivos
    archivoN.close()
    archivoP.close()


separarPosNeg()
