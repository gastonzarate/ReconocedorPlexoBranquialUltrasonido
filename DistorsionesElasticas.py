# Import stuff
import numpy as np
import cv2
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import shutil
import os
import re
from skimage import img_as_ubyte,io

id = "Z"

def elastic_transform(image, alpha, sigma, alpha_affine, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape[:2]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0] + square_size, center_square[1] - square_size],
                       center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

    return map_coordinates(image, indices, order=1, mode='reflect').reshape(shape)

# Define function to draw a grid
def draw_grid(im, grid_size):
    # Draw grid lines
    for i in range(0, im.shape[1], grid_size):
        cv2.line(im, (i, 0), (i, im.shape[0]), color=(255,))
    for j in range(0, im.shape[0], grid_size):
        cv2.line(im, (0, j), (im.shape[1], j), color=(255,))


"""
im = cv2.imread("DatosNormalizados/train/1_1.png", 0)
im_mask = cv2.imread("DatosNormalizados/train/1_1_mask.png", 0)

# Draw grid lines
draw_grid(im, 50)
draw_grid(im_mask, 50)

im_merge = np.concatenate((im[...,None], im_mask[...,None]), axis=2)
im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08, im_merge.shape[1] * 0.08)

# Split image and mask
im_t = im_merge_t[...,0]
im_mask_t = im_merge_t[...,1]

# Display result
cv2.imshow('image',im_t)
cv2.imshow('mask',im_mask_t)
cv2.waitKey(0)
"""
lista_img = []
rutaCarpeta = os.path.join("DatosNormalizados","Distorsiones")
rutaOrigen = os.path.join("DatosNormalizados","train")
rutaPos = os.path.join("Datos","lista-pos.txt")

def listarImagenes():
    archivo = open(rutaPos,"r")
    for name in archivo.readlines():
        lista_img.append(name)
    return lista_img

def crearDirectorios():
    op = "S"
    # Pregunta si existen los directorios o archivos para eliminarlos
    if os.path.exists(rutaCarpeta):
        op = raw_input("Esta seguro que desea eliminar el contenido de la carpeta Distorsiones(S/n)")
        if op =="S":
            shutil.rmtree(rutaCarpeta, ignore_errors=True)
            os.makedirs(rutaCarpeta)
    else:
        os.makedirs(rutaCarpeta)

def generarConjunto():
    #lista solo las imagenes positivas
    lista = listarImagenes()
    crearDirectorios()
    for name in lista:
        nomArch = name.split('.')[0]
        mascara = nomArch + "_mask.png"
        im_mask = cv2.imread(os.path.join(rutaOrigen,mascara), 0)
        im = cv2.imread(os.path.join(rutaOrigen,nomArch+".png"),0)

        #Genera 5 imagenes distorsionadas por cada imagen positiva
        for i in range(1,6):
            #le aplica la misma distorsion a la imagen y a su correspondiente mascara
            im_merge = np.concatenate((im[..., None], im_mask[..., None]), axis=2)
            im_merge_t = elastic_transform(im_merge, im_merge.shape[1] * 2, im_merge.shape[1] * 0.08,
                                        im_merge.shape[1] * 0.08)
            #hace un split para reconstruir las dos imagenes
            im_t = im_merge_t[..., 0]
            im_mask_t = im_merge_t[..., 1]
            nomImg = nomArch+'_'+ id +str(i)+'.png'
            nomMask = nomArch+'_mask_' + id + str(i)+'.png'
            cv2.imwrite(os.path.join(rutaCarpeta,nomImg),im_t)
            cv2.imwrite(os.path.join(rutaCarpeta,nomMask), im_mask_t)
            print 'Grabado im: '+nomImg
            print 'Grabado mask:'+nomMask

generarConjunto()
