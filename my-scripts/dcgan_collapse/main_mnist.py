#!/usr/bin/env python3
import sys
sys.path.insert(0,'/usr/local/lib')
# ln -s /usr/local/lib/libplaidml.dylib /usr/local/Cellar/python/3.7.0/Frameworks/Python.framework/Versions/3.7/lib/libplaidml.dylib
# ln -s /usr/local/lib/libplaidml.dylib /usr/local/Cellar/python/3.6.5_1/Frameworks/Python.framework/Versions/3.6/lib/libplaidml.dylib

# print(sys.path)
# import plaidml.keras
# plaidml.keras.install_backend()

from dcgan_28x28x1 import *



import math
import numpy as np

import matplotlib.pyplot as plt
import cv2

def save_images(imgs, index, dir_path):
    B, H, W, C = imgs.shape
    batch= imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    w_num = np.ceil(np.sqrt(B)).astype(np.int)
    h_num = int(np.ceil(B / w_num))
    out = np.zeros((h_num*H, w_num*W), dtype=np.uint8)
    for i in range(B):
        x = i % w_num
        y = i // w_num
        out[y*H:(y+1)*H, x*W:(x+1)*W] = batch[i, ..., 0]
    cv2.imshow("test",out)
    cv2.waitKey(1);


import os
from keras.datasets import mnist
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from PIL import Image

discriminator = discriminator_model()


BATCH_SIZE = 64
NUM_EPOCH = 20

def train():
    (X_train, y_train), (_, _) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5)/127.5
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])

    discriminator = discriminator_model()
    d_opt = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0001)
    discriminator.compile(loss='binary_crossentropy', optimizer=d_opt)

    discriminator.trainable = False
    generator = generator_model()
    dcgan = Sequential([generator, discriminator])
    g_opt = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0001)
    dcgan.compile(loss='binary_crossentropy', optimizer=g_opt)

    num_batches = int(X_train.shape[0] / BATCH_SIZE)
    print('Number of batches:', num_batches)
    print("-------------------------------------");
    counter = 0
    for epoch in range(NUM_EPOCH):

        for index in range(num_batches):
            counter += 1
            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE].reshape(BATCH_SIZE,28,28,1)
            generated_images = generator.predict(noise, verbose=0)

            save_images(generated_images,0,".")
            #if counter%10 == 0: save_images(generated_images,0,".")

            print(image_batch.shape,generated_images.shape,flush=True)

            X = np.concatenate((image_batch, generated_images))
            y = [1]*BATCH_SIZE + [0]*BATCH_SIZE
            d_loss = discriminator.train_on_batch(X, y)

            noise = np.array([np.random.uniform(-1, 1, 100) for _ in range(BATCH_SIZE)])
            g_loss = dcgan.train_on_batch(noise, [1]*BATCH_SIZE)
            print("epoch: %d, batch: %d, g_loss: %f, d_loss: %f" % (epoch, index, g_loss, d_loss),flush=True)

        generator.save_weights('generator_mnist.h5')
        discriminator.save_weights('discriminator_mnist.h5')


train()
