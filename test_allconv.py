from __future__ import print_function

import sys

import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Activation, Convolution2D, GlobalAveragePooling2D, merge
from keras.utils import np_utils
from keras.optimizers import SGD
from keras import backend as K
from keras.models import Model, load_model
from keras.layers.core import Lambda
from keras.callbacks import ModelCheckpoint
import pandas
import cv2
import numpy as np

modelfile = sys.argv[1] if len(sys.argv) > 1 else 'weights.hdf5'

K.set_image_dim_ordering('tf')


(X_train, y_train), (X_test, y_test) = cifar10.load_data()
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print(X_test.shape[0], 'test samples')
#print (X_train.shape[1:])

nb_classes = 10
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = load_model(modelfile)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

e = model.evaluate(X_test, Y_test, batch_size=32)
print("Loss: %0.3e, Accuarcy: %0.3f" % (e[0], e[1]))
