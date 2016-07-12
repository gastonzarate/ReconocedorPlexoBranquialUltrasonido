import os
import cv2
import re
import ReducirRuido

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

def buscarImagenesIguales(imgs):
    lista = []
    for i in range(len(imgs)):
        for j in range(len(imgs)):
            comp = compararImagenes(imgs[i],imgs[j])
            if(comp==True):
              lista.append("%i - %i",(i,j))
            if(i==j):
                break
    return lista

def compararImagenes(a,b):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3), (1, 1))
    img = cv2.subtract(a, b)
    img,thresh = cv2.threshold(img, 15, 255, cv2.CV_THRESH_BINARY)
    img = cv2.morphologyEx(img, cv2.CV_MOP_OPEN, kernel, (-1, -1), 1)
    im2, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if(len(contours)==0):
        bandera = True
    else:
        bandera = False
    return bandera

#Carga las imagenes en memoria
imgs = []
mask = []
for base,dirs,files in os.walk('train2'):
    files.sort(cmp=compara)
    for name in files:
        print os.path.join(base, name)
        i = cv2.imread(os.path.join(base, name),cv2.IMREAD_GRAYSCALE)

        #i = ReducirRuido.filtroMorfologico(i,iterations=5)
        #i = ReducirRuido.denoiseNonLocalMeans(i)
        #cv2.imshow('image', i)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        if 'mask' in name:
            mask.append(i)
        else:
            imgs.append(i)
    lista = buscarImagenesIguales(imgs)
    print lista

