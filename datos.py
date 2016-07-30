from __future__ import print_function

import os
import numpy as np
import compara as c
import cv2

datos_path = ''
destino_train_path = ''
destino_test_path = ''

im_filas = 420
im_columnas = 580

def set_paths(origen_datos, destino_train, destino_test):
    global datos_path
    global destino_train_path
    global destino_test_path
    datos_path = origen_datos
    destino_train_path = destino_train
    destino_test_path = destino_test

def crear_datos_entrenamiento():
    train_data_path = os.path.join(datos_path, 'train')
    imagenes = os.listdir(train_data_path)
    imagenes.sort(cmp=c.compara)
    total = len(imagenes) / 2

    imgs = np.ndarray((total, 1, im_filas, im_columnas), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, im_filas, im_columnas), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creando imagenes de entrenamiento...')
    print('-' * 30)

    for imagen in imagenes:
        if 'mask' in imagen:
            continue
        nombre_mask = imagen.split('.')[0] + '_mask.tif'
        img = cv2.imread(os.path.join(train_data_path, imagen), cv2.IMREAD_GRAYSCALE)
        img_mask = cv2.imread(os.path.join(train_data_path, nombre_mask), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Hecho: {0}/{1} imagenes'.format(i, total))
        i += 1
    print('Carga finalizada.')

    print('Guardando train en .npy')
    np.save(destino_train_path+'imgs_train.npy', imgs)
    np.save(destino_train_path+'imgs_mask_train.npy', imgs_mask)
    print('Guardado de train en .npy finalizado.')

def cargar_datos_entrenamiento():
    imgs_train = np.load(datos_path+'imgs_train.npy')
    imgs_mask_train = np.load(datos_path+'imgs_mask_train.npy')
    return imgs_train, imgs_mask_train

def crear_datos_test():
    train_data_path = os.path.join(datos_path, 'test')
    imagenes = os.listdir(train_data_path)
    imagenes.sort(cmp=c.compara)
    total = len(imagenes)

    imgs = np.ndarray((total, 1, im_filas, im_columnas), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print('-'*30)
    print('Creando imagenes de test...')
    print('-'*30)
    for imagen in imagenes:
        img_id = int(imagen.split('.')[0])
        img = cv2.imread(os.path.join(train_data_path, imagen), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Hecho: {0}/{1} imagenes'.format(i, total))
        i += 1
    print('Carga finalizada.')

    print('Guardando test en .npy')
    np.save(destino_test_path+'imgs_test.npy', imgs)
    np.save(destino_test_path+'imgs_id_test.npy', imgs_id)
    print('Guardado de test en .npy finalizado.')

def cargar_datos_test():
    imgs_test = np.load(datos_path+'imgs_test.npy')
    imgs_id = np.load(datos_path+'imgs_id_test.npy')
    return imgs_test, imgs_id

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], im_filas, im_columnas), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (im_columnas, im_filas), interpolation=cv2.INTER_CUBIC)
    return imgs_p

def crear_datos_sd(marcadas):
    train_data_path = os.path.join(datos_path, 'train')
    imagenes = os.listdir(train_data_path)
    imagenes.sort(cmp=c.compara)
    total = len(imagenes) / 2 - len(marcadas)
    total_marcadas = len(marcadas)

    imgs = np.ndarray((total, 1, im_filas, im_columnas), dtype=np.uint8)
    imgs_mask = np.ndarray((total, 1, im_filas, im_columnas), dtype=np.uint8)

    i = 0
    print('-' * 30)
    print('Creando imagenes de entrenamiento...')
    print('-' * 30)

    for imagen in imagenes:
        if marcadas.contains(i):
            continue
        else:
            if 'mask' in imagen:
                continue
            nombre_mask = imagen.split('.')[0] + '_mask.tif'
            img = cv2.imread(os.path.join(train_data_path, imagen), cv2.IMREAD_GRAYSCALE)
            img_mask = cv2.imread(os.path.join(train_data_path, nombre_mask), cv2.IMREAD_GRAYSCALE)

            img = np.array([img])
            img_mask = np.array([img_mask])

            imgs[i] = img
            imgs_mask[i] = img_mask

            if i % 100 == 0:
                print('Hecho: {0}/{1} imagenes'.format(i, total))
        i += 1
    print('Carga finalizada.')

    print('Guardando train sd en .npy')
    np.save(destino_train_path+'imgs_train_sd.npy', imgs)
    np.save(destino_train_path+'imgs_mask_train_sd.npy', imgs_mask)
    print('Guardado de train sd en .npy finalizado.')



