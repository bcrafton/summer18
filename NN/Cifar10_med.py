
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

import time
import tensorflow as tf
import keras
import math
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
from Activation import Tanh
from Activation import LeakyRelu

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 25
ALPHA = 1e-4

##############################################

EPSILON = 0.12

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
XTRAIN = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTRAIN = tf.placeholder(tf.float32, [None, 10])

XTEST = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTEST = tf.placeholder(tf.float32, [None, 10])

sqrt_fan_in = math.sqrt(32 * 32 * 3)
W0 = tf.Variable(tf.random_uniform(shape=[3, 3, 3, 32], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[3, 3, 3, 32], num_classes=10, filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False)

sqrt_fan_in = math.sqrt(32 * 32 * 32)
W1 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 32], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l1 = Convolution(input_sizes=[batch_size, 32, 32, 32], filter_sizes=[3, 3, 32, 32], num_classes=10, filters=W1, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False)

l2 = MaxPool(size=[batch_size, 32, 32, 32], stride=[1, 2, 2, 1])

#l3 = Dropout(rate=0.25)

sqrt_fan_in = math.sqrt(16 * 16 * 32)
W4 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 64], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l4 = Convolution(input_sizes=[batch_size, 16, 16, 32], filter_sizes=[3, 3, 32, 64], num_classes=10, filters=W4, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False)

sqrt_fan_in = math.sqrt(16 * 16 * 64)
W5 = tf.Variable(tf.random_uniform(shape=[3, 3, 64, 64], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l5 = Convolution(input_sizes=[batch_size, 16, 16, 64], filter_sizes=[3, 3, 64, 64], num_classes=10, filters=W5, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False)

l6 = MaxPool(size=[batch_size, 16, 16, 64], stride=[1, 2, 2, 1])

#l7 = Dropout(rate=0.25)

l8 = ConvToFullyConnected(shape=[8, 8, 64])

sqrt_fan_in = math.sqrt(8 * 8 * 64)
W9 = tf.Variable(tf.random_uniform(shape=[8*8*64, 512], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l9 = FullyConnected(size=[8*8*64, 512], num_classes=10, weights=W9, alpha=ALPHA, activation=Tanh(), last_layer=False)

#l10 = Dropout(rate=0.5)

sqrt_fan_in = math.sqrt(512)
W11 = tf.Variable(tf.random_uniform(shape=[512, 10], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l11 = FullyConnected(size=[512, 10], num_classes=10, weights=W11, alpha=ALPHA, activation=Tanh(), last_layer=True)

##############################################

model = Model(layers=[l0, l1, l2, l4, l5, l6, l8, l9, l11])

predict = model.predict(X=XTEST)

grads_and_vars = model.dfa(X=XTRAIN, Y=YTRAIN)
optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA, beta1=0.9, beta2=0.999, epsilon=0.1).apply_gradients(grads_and_vars=grads_and_vars)
#optimizer = tf.train.GradientDescentOptimizer(learning_rate=ALPHA).apply_gradients(grads_and_vars=grads_and_vars)
#optimizer = tf.train.MomentumOptimizer(learning_rate=ALPHA, momentum=0.99).apply_gradients(grads_and_vars=grads_and_vars)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
correct_prediction_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
    print (ii)
    for jj in range(0, TRAIN_EXAMPLES, BATCH_SIZE):
        start = jj % TRAIN_EXAMPLES
        end = jj % TRAIN_EXAMPLES + BATCH_SIZE
        sess.run([grads_and_vars, optimizer], feed_dict={batch_size: BATCH_SIZE, XTRAIN: x_train[start:end], YTRAIN: y_train[start:end]})

# print(sess.run(accuracy, feed_dict={batch_size: TEST_EXAMPLES, XTEST: x_test, YTEST: y_test}))

count = 0
total_correct = 0

for ii in range(0, TEST_EXAMPLES, BATCH_SIZE):
    start = ii % TEST_EXAMPLES
    end = ii % TEST_EXAMPLES + BATCH_SIZE
    correct = sess.run(correct_prediction_sum, feed_dict={batch_size: BATCH_SIZE, XTEST: x_test[start:end], YTEST: y_test[start:end]})

    count += BATCH_SIZE
    total_correct += correct

    print (count)
    print (total_correct)
    print (total_correct * 1.0 / count)



