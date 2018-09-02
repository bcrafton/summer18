
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 100
TRAINING_EXAMPLES = 50000
TESTING_EXAMPLES = 10000
learning_rate = 5e-3

cifar10 = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAINING_EXAMPLES, 3072)
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TESTING_EXAMPLES, 3072)
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Dense(1000, activation='relu', input_shape=(3072,)))
model.add(Dense(1000, activation='relu'))
model.add(Dense(1000, activation='relu'))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, 
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
