
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

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

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = 1
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 25
ALPHA = 1e-3

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform(shape=[3, 3, 3, 32]) * (2 * 0.0025) - 0.0025)
l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[3, 3, 3, 32], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l1 = MaxPool(size=[batch_size, 32, 32, 32], stride=[1, 2, 2, 1])

W2 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 32]) * (2 * 0.0025) - 0.0025)
l2 = Convolution(input_sizes=[batch_size, 16, 16, 32], filter_sizes=[3, 3, 32, 32], filters=W2, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l3 = MaxPool(size=[batch_size, 16, 16, 32], stride=[1, 2, 2, 1])

W4 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 64]) * (2 * 0.0025) - 0.0025)
l4 = Convolution(input_sizes=[batch_size, 8, 8, 32], filter_sizes=[3, 3, 32, 64], filters=W4, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l5 = MaxPool(size=[batch_size, 8, 8, 64], stride=[1, 2, 2, 1])

l6 = ConvToFullyConnected(shape=[4, 4, 64])

W7 = tf.Variable(tf.random_uniform(shape=[4*4*64, 128]) * (2 * 0.0025) - 0.0025)
l7 = FullyConnected(size=[4*4*64, 128], weights=W7, alpha=ALPHA, activation=Relu(), last_layer=False)

l8 = Dropout(rate=0.5)

W9 = tf.Variable(tf.random_uniform(shape=[128, 10]) * (2 * 0.0025) - 0.0025)
l9 = FullyConnected(size=[128, 10], weights=W9, alpha=ALPHA, activation=Relu(), last_layer=False)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8])

##############################################

ret = model.train(X=X, Y=Y)
predict = model.predict(X=X)

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(Y, 1), predictions=tf.argmax(predict, 1))

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

for ii in range(EPOCHS):
    for jj in range(0, TRAIN_EXAMPLES, BATCH_SIZE):
        print (str(ii * TRAIN_EXAMPLES + jj) + "/" + str(EPOCHS * TRAIN_EXAMPLES))

        start = jj % TRAIN_EXAMPLES
        end = jj % TRAIN_EXAMPLES + BATCH_SIZE
        sess.run([ret], feed_dict={batch_size: BATCH_SIZE, X: x_train[start:end], Y: y_train[start:end]})

for jj in range(0, TEST_EXAMPLES, BATCH_SIZE):
    start = jj % TEST_EXAMPLES
    end = jj % TEST_EXAMPLES + BATCH_SIZE
    print('acc:', sess.run(acc_op, feed_dict={batch_size: BATCH_SIZE, X: x_test[start:end], Y: y_test[start:end]}))













