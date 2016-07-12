import numpy as np
import cv2
from skimage import img_as_float
from skimage.restoration import denoise_nl_means,inpaint,denoise_tv_chambolle, denoise_bilateral,denoise_tv_bregman


def filtroMorfologico(imagen,iterations=1):
    # Crear un kernel de '1' de 3x3
    kernel = np.ones((3, 3), np.uint8)

    # Se aplica la transformacion: Morphological Gradient
    transformacion = cv2.dilate(imagen, kernel, iterations) - cv2.erode(imagen, kernel, iterations)

    return transformacion

def denoiseNonLocalMeans(imagen):
    """
    Reemplaza la intensidad de cada pixel con la media de los pixels a su alrededor
    """
    noisy = img_as_float(imagen)

    denoise = denoise_nl_means(noisy, 7, 9, 0.08)

    return denoise

def denoiseBilateral(imagen,multichannel):
    """
    -Reemplaza el valor de cada pixel en funcion de la proximidad espacial y radiometrica
     medida por la funcion Gaussiana de la distancia euclidiana entre dos pixels y con
     cierta desviacion estandar.
    -False si la imagen es una escala de grises, sino True
    """
    noisy = img_as_float(imagen)

    denoise = denoise_bilateral(noisy, 7, 9, 0.08,multichannel)

    return denoise


def denoiseTV_Chambolle(imagen,multichannel):
    """
    -Tiende a producir imagenes como las de los dibujos animados.
    -Reduce al minimo la variacion total de la imagen
    """
    noisy = img_as_float(imagen)

    denoise = denoise_tv_chambolle(noisy, 7, 9, 0.08,multichannel)

    return denoise

def denoiseTV_Bregman(imagen,isotropic):
    """
    -isotropic es el atributo para cambiar entre filtrado isotropico y anisotropico
    """
    noisy = img_as_float(imagen)

    denoise = denoise_tv_bregman(noisy, 7, 9, 0.08, isotropic)

    return denoise

def denoiseInpaint(imagen,mask,multichannel):
    noisy = img_as_float(imagen)
    image_orig = noisy

    # Afecta a la imagen original en las regiones marcadas por la mascara
    # Se puede cambiar de acuerdo a lo que se necesite
    # En este caso produce defectos en la zona
    image_defect = image_orig.copy()
    for layer in range(image_defect.shape[-1]):
        image_defect[np.where(mask)] = 1

    image_result = inpaint.inpaint_biharmonic(image_defect, mask, multichannel)

    return image_result
