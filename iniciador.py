import os
import cv2
import re

#Funcion para ordenar los nombres de las imagenes
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

#Carga las imagenes en memoria
imgs = []
mask = []
for base,dirs,files in os.walk('train'):
    files.sort(cmp=compara)
    for name in files:
        print os.path.join(base, name)
        i = cv2.imread(os.path.join(base, name),cv2.IMREAD_GRAYSCALE)
        if 'mask' in files:
            mask.append(i)
        else:
            imgs.append(i)


