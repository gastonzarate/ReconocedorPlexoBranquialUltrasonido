import os
import cv2
import ReducirRuido
from skimage import img_as_ubyte


rutaOrigen = os.path.join("Datos","train")
print os._exists(rutaOrigen)
for base, dirs, files in os.walk(rutaOrigen):
    for name in files:
        img = cv2.imread(os.path.join(rutaOrigen, name))
        nomArch = name.split('.')[0]
        bordes = cv2.Canny(img, 100, 200)

        cv2.imshow(nomArch,img)
        cv2.imshow(nomArch + ": Bordes", bordes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

#        img = img_as_ubyte(img)
 #       bordes = cv2.Canny(img, 100, 200)
  #      cv2.imshow(nomArch + " : CV_U8", img)
   #     cv2.imshow(nomArch + ": Bordes", bordes)
    #    cv2.waitKey(0)
     #   cv2.destroyAllWindows()

        img = ReducirRuido.denoiseMorfologico(img)
        bordes = cv2.Canny(img, 100, 200)
        cv2.imshow(nomArch + ":Morfologico", img)
        cv2.imshow(nomArch + ": Bordes", bordes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        img = ReducirRuido.denoiseNonLocalMeans(img)
        img = img_as_ubyte(img)

        bordes = cv2.Canny(img, 100, 200)
        cv2.imshow(nomArch+" : NonLocalMeans", img)
        cv2.imshow(nomArch + ": Bordes", bordes)
        cv2.waitKey(0)
        cv2.destroyAllWindows(0)


