from __future__ import print_function

import argparse

import keras
from keras import backend as K
import numpy as np


def preprocess(x):
    x = x.astype('float32')
    x /= 255
    x -= 0.5
    return x


def quantize(x, bits):
    levels = 2**bits
    max_x = np.abs(x).max()
    alpha = 0.5 * levels / max_x
    y = np.round(alpha * x) / alpha
    return y


parser = argparse.ArgumentParser(description="Test allconv CIFAR-10 network")
parser.add_argument('--quantize', action='store_true')
parser.add_argument('modelfile', nargs='?', default='weights.hdf5', help="Model file to load")
args = parser.parse_args()

K.set_image_dim_ordering('tf')

(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train, X_test = preprocess(X_train), preprocess(X_test)

nb_classes = 10
Y_train = keras.utils.np_utils.to_categorical(y_train, nb_classes)
Y_test = keras.utils.np_utils.to_categorical(y_test, nb_classes)

model = keras.models.load_model(args.modelfile)

if args.quantize:
    for layer in model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            w, b = layer.get_weights()
            print(w[:, :, 0, 0])
            w = quantize(w, 8)
            print(w[:, :, 0, 0])
            layer.set_weights((w, b))

e = model.evaluate(X_test, Y_test, batch_size=32)
print("Loss: %0.3e, Accuarcy: %0.3f" % (e[0], e[1]))
