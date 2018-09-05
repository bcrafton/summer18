from __future__ import print_function

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--verbose', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = args.epochs
TRAINING_EXAMPLES = 60000
TESTING_EXAMPLES = 10000
learning_rate = args.alpha

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAINING_EXAMPLES, 28, 28, 1)
x_test = x_test.reshape(TESTING_EXAMPLES, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), padding="same", activation='tanh', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(64, (3, 3), padding="same", activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Conv2D(64, (3, 3), padding="same", activation='tanh'))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Flatten())
model.add(Dense(512, activation='tanh'))
model.add(Dense(128, activation='tanh'))
model.add(Dense(NUM_CLASSES, activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, 
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=args.verbose,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
