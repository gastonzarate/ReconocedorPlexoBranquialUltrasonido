import os
import numpy as np
import cv2
import shutil
import compara as c

rutaCarpeta = "Subida"
ruta = os.path.join(rutaCarpeta, "submission.csv")
rutaImg = os.path.join("DatosNormalizados", "test")
clasificador = os.path.join("Clasificadores",'cascade5_lbp.xml')
ancho = 580
alto = 420
margen = 5


# Si existe el directorio carpeta la elimina
if os.path.exists(rutaCarpeta):
    shutil.rmtree(rutaCarpeta, ignore_errors=True)
os.makedirs(rutaCarpeta)
# Crea el archivo de subida
archivo = open(ruta, 'w')
archivo.close()
# Abre el archivo de subida
archivo = open(ruta, 'a')
archivo.write("img,pixels\n")


#Carga el clasificador
cascade = cv2.CascadeClassifier(clasificador)

#Recorre las imagenes, las clasifica y guarda su solucion
for base, dirs, files in os.walk(rutaImg):
    files.sort(cmp=c.compara)
    for name in files:
        if not 'mask' in name:

            #Carga la imagen
            img = cv2.imread(os.path.join(base, name))

            #Recorta la imagen
            #img = img[60,50:460:320]

            #Detecta las imagenes
            pa = cascade.detectMultiScale(
                 img,
                 scaleFactor=1.01,
                 minNeighbors=5,
                 minSize=(5, 5),
                 maxSize=(300, 300),
                 flags = 0)
            #Guarda en el archivo su numero
            archivo.write(name.split('.')[0] + ",")
            bandera = False
            #Guarda en el archivo la primera deteccion
            for (x, y, w, h) in pa:
                 x = x + margen
                 y = y + margen
                 h = h - margen*2
                 w = w - margen*2
                 com = (x - 1) * ancho + y
                 fin = ((x - 1) + w) * ancho + y
                 for i in range(com, fin, alto):
                     archivo.write('%d %d ' % (i,h))
                 print name+":Detectado %d" %len(pa)
                 bandera = True
                 break
            if not bandera:
                print name + ":No Detectado"
                bandera=False

            archivo.write("\n")
archivo.close()






