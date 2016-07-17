import os
import cv2
import re
import ReducirRuido
import numpy as np

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

def separarImagenes():
    # Rutas de los archivos y carpetas a crear
    rutaPos = "pos"
    rutaNeg = "neg"
    archPos = "pos.info"
    archNeg = "neg.info"
    #Pregunta si existen los directorios o archivos para eliminarlos
    if os.path.exists(rutaPos):
        os.rmdir(rutaPos)
    if os.path.exists(rutaNeg):
        os.rmdir(rutaNeg)
    if os.path.exists(archPos):
        os.rmdir(archPos)
    if os.path.exists(archNeg):
        os.rmdir(archNeg)

    #Crea las carpetas
    os.makedirs(rutaPos)
    os.makedirs(rutaNeg)
    #Crea los archivos
    archivoP = open(archPos, 'w')
    archivoP.close()
    archivoN = open(archNeg, 'w')
    archivoN.close()
    #Abre los archivos para escribir en ellos
    archivoP = open(archPos, 'a')
    archivoN = open(archNeg, 'a')

    #Recorre la carpetas de entrenamiento
    iNeg = 1
    iPos = 1
    for base, dirs, files in os.walk('train'):
        files.sort(cmp=compara)
        for name in files:
            print os.path.join(base, name)
            #Leo la imagen
            img = cv2.imread(os.path.join(base, name), cv2.IMREAD_GRAYSCALE)

            if 'mask' in name:
                #Al ser la mascara, su imagen correspondiente fue cargada en el ciclo anterior
                mask = img
                #Se detectan los contornos
                mask, thresh = cv2.threshold(mask, 15, 255, cv2.THRESH_BINARY)
                im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                if(len(contours)==0):
                    #Si no se encontro contorno se la almacena como negativa
                    cv2.imwrite(os.path.join(rutaNeg,iNeg+".png"),imgAnt)
                    archivoN.write(os.path.join(rutaNeg,iNeg+".png")+'\n')
                    iNeg = iNeg + 1
                else:
                    #Al encontrarse contorno se busca el de mayor area
                    areas = [cv2.contourArea(c) for c in contours]
                    max_index = np.argmax(areas)
                    cnt = contours[max_index]
                    #Se obtiene la recta correspondiente al contorno
                    x, y, w, h = cv2.boundingRect(cnt)
                    #Se almacena como negativa
                    cv2.imwrite(os.path.join(rutaPos, iPos + ".png"), imgAnt)
                    archivoP.write(os.path.join(rutaPos,iPos+".png")+' 1 %d %d %d %d \n' %(x,y,h,w))
                    iPos = iPos + 1
            else:
                # La normalizo
                imgAnt = ReducirRuido.filtroMorfologico(img, iterations=5)
                imgAnt = ReducirRuido.denoiseNonLocalMeans(imgAnt)
    #Se cierran los archivos
    archivoN.close()
    archivoP.close()