import os
import cv2
import ReducirRuido

ruta = "submission.csv"
# Pregunta si existen los directorios o archivos para eliminarlos
if os.path.exists(ruta):
    os.rmdir(ruta)

# Crea los archivos
archivo = open(ruta, 'w')
archivo.close()
# Abre los archivos para escribir en ellos
archivo = open(ruta, 'a')
archivo.write("img,pixels")

ancho = 580
alto = 420
numImg = 1

cascade = cv2.CascadeClassifier('cascade2.xml')
for base, dirs, files in os.walk('test'):
    for name in files:
        if not 'mask' in name:
            print numImg
            img = cv2.imread(os.path.join(base, name))
            img = ReducirRuido.denoiseMorfologico(img)
            img = ReducirRuido.denoiseNonLocalMeans(img)
            img = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=5)
            pa = cascade.detectMultiScale(
                img,
                scaleFactor=1.01,
                minNeighbors=9,
                minSize=(10, 10),
                maxSize=(200, 200),
                flags=0)
            archivo.write("%d," % numImg)
            for (x, y, w, h) in pa:
                com = (x - 1) * ancho + y
                fin = ((x - 1) + w) * ancho + y
                for i in range(com, fin, alto):
                    hasta = i + h
                    archivo.write('%d %d ' % (i,hasta))
                break
            archivo.write("\n")
            numImg = numImg + 1
archivo.close()



