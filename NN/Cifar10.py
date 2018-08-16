
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(2)

import time
import tensorflow as tf
import keras
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution
from MaxPool import MaxPool
from Dropout import Dropout

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = 1
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 20
ALPHA = 1e-2

##############################################

EPSILON = 0.12

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform(shape=[5, 5, 3, 96]) * 2 * EPSILON - EPSILON)
l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l1 = Dropout(rate=0.25)

W2 = tf.Variable(tf.random_uniform(shape=[5, 5, 96, 128]) * 2 * EPSILON - EPSILON)
l2 = Convolution(input_sizes=[batch_size, 32, 32, 96], filter_sizes=[5, 5, 96, 128], filters=W2, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l3 = Dropout(rate=0.25)

l4 = MaxPool(size=[batch_size, 32, 32, 128], stride=[1, 2, 2, 1])

W5 = tf.Variable(tf.random_uniform(shape=[5, 5, 128, 256]) * 2 * EPSILON - EPSILON)
l5 = Convolution(input_sizes=[batch_size, 16, 16, 128], filter_sizes=[5, 5, 128, 256], filters=W5, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l6 = Dropout(rate=0.5)

l7 = MaxPool(size=[batch_size, 16, 16, 256], stride=[1, 2, 2, 1])

l8 = ConvToFullyConnected(shape=[8, 8, 256])

W9 = tf.Variable(tf.random_uniform(shape=[8*8*256, 2048]) * 2 * EPSILON - EPSILON)
l9 = FullyConnected(size=[8*8*256, 2048], weights=W9, alpha=ALPHA, activation=Relu(), last_layer=False)

l10 = Dropout(rate=0.5)

W11 = tf.Variable(tf.random_uniform(shape=[2048, 10]) * 2 * EPSILON - EPSILON)
l11 = FullyConnected(size=[2048, 10], weights=W11, alpha=ALPHA, activation=Relu(), last_layer=False)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11])

predict = model.predict(X=X)

ret = model.train(X=X, Y=Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, 32, 32, 3)
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, 10)

for ii in range(0, EPOCHS * TRAIN_EXAMPLES, BATCH_SIZE):
    start = ii % TRAIN_EXAMPLES
    end = ii % TRAIN_EXAMPLES + BATCH_SIZE
    sess.run([ret], feed_dict={batch_size: BATCH_SIZE, X: x_train[start:end], Y: y_train[start:end]})

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

count = 0
total_acc = 0

for ii in range(0, TEST_EXAMPLES, BATCH_SIZE):
    start = ii % TEST_EXAMPLES
    end = ii % TEST_EXAMPLES + BATCH_SIZE
    acc = sess.run(accuracy, feed_dict={batch_size: BATCH_SIZE, X: x_test[start:end], Y: y_test[start:end]})

    count += 1
    total_acc += acc









