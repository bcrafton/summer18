
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=5e-5)
parser.add_argument('--gpu', type=int, default=-1)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

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
from Activation import Softmax
from Activation import LeakyRelu

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size
ALPHA = args.alpha

##############################################

sparse = args.sparse

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
XTRAIN = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTRAIN = tf.placeholder(tf.float32, [None, 10])
XTRAIN = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), XTRAIN)

XTEST = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTEST = tf.placeholder(tf.float32, [None, 10])
XTEST = tf.map_fn(lambda frame1: tf.image.per_image_standardization(frame1), XTEST)

W0 = tf.Variable(tf.zeros(shape=[5, 5, 3, 96]))
sqrt_fan_in = math.sqrt(32 * 32 * 3)
#W0 = tf.Variable(tf.random_uniform(shape=[5, 5, 3, 96], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], num_classes=10, filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

#l1 = Dropout(rate=0.25)

W2 = tf.Variable(tf.zeros(shape=[5, 5, 96, 128]))
sqrt_fan_in = math.sqrt(32 * 32 * 96)
#W2 = tf.Variable(tf.random_uniform(shape=[5, 5, 96, 128], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l2 = Convolution(input_sizes=[batch_size, 32, 32, 96], filter_sizes=[5, 5, 96, 128], num_classes=10, filters=W2, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

#l3 = Dropout(rate=0.25)

l4 = MaxPool(size=[batch_size, 32, 32, 128], stride=[1, 2, 2, 1])

W5 = tf.Variable(tf.zeros(shape=[5, 5, 128, 256]))
sqrt_fan_in = math.sqrt(16 * 16 * 128)
#W5 = tf.Variable(tf.random_uniform(shape=[5, 5, 128, 256], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l5 = Convolution(input_sizes=[batch_size, 16, 16, 128], filter_sizes=[5, 5, 128, 256], num_classes=10, filters=W5, stride=1, padding=1, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

#l6 = Dropout(rate=0.5)

l7 = MaxPool(size=[batch_size, 16, 16, 256], stride=[1, 2, 2, 1])

l8 = ConvToFullyConnected(shape=[8, 8, 256])

W9 = tf.Variable(tf.zeros(shape=[8*8*256, 2048]))
sqrt_fan_in = math.sqrt(8 * 8 * 256)
#W9 = tf.Variable(tf.random_uniform(shape=[8*8*256, 2048], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l9 = FullyConnected(size=[8*8*256, 2048], num_classes=10, weights=W9, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

#l10 = Dropout(rate=0.5)

W11 = tf.Variable(tf.zeros(shape=[2048, 2048]))
sqrt_fan_in = math.sqrt(2048)
#W11 = tf.Variable(tf.random_uniform(shape=[2048, 2048], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l11 = FullyConnected(size=[2048, 2048], num_classes=10, weights=W11, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

#l12 = Dropout(rate=0.5)

W13 = tf.Variable(tf.zeros(shape=[2048, 10]))
sqrt_fan_in = math.sqrt(2048)
#W13 = tf.Variable(tf.random_uniform(shape=[2048, 10], minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
l13 = FullyConnected(size=[2048, 10], num_classes=10, weights=W13, alpha=ALPHA, activation=Tanh(), last_layer=True, sparse=sparse)

#model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l11, l12, l13])
model = Model(layers=[l0, l2, l4, l5, l7, l8, l9, l11, l13])

predict = model.predict(X=XTEST)

if args.dfa:
    grads_and_vars = model.dfa(X=XTRAIN, Y=YTRAIN)
else:
    grads_and_vars = model.train(X=XTRAIN, Y=YTRAIN)
    
optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA, beta1=0.9, beta2=0.999, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
correct_prediction_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, 32, 32, 3)
y_test = keras.utils.to_categorical(y_test, 10)

##############################################

filename = "large_" + str(args.epochs) + "_" + str(args.batch_size) + "_" + str(args.alpha) + "_" + str(args.dfa) + "_" + str(args.sparse) + ".results"
f = open(filename, "w")
f.write(filename + "\n")
f.close()

##############################################

for ii in range(EPOCHS):
    print (ii)
    for jj in range(0, int(TRAIN_EXAMPLES/BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        start = jj % TRAIN_EXAMPLES
        end = jj % TRAIN_EXAMPLES + BATCH_SIZE
        sess.run([grads_and_vars, optimizer], feed_dict={batch_size: BATCH_SIZE, XTRAIN: x_train[start:end], YTRAIN: y_train[start:end]})
    
    count = 0
    total_correct = 0
    
    for jj in range(0, int(TEST_EXAMPLES/BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        start = jj % TEST_EXAMPLES
        end = jj % TEST_EXAMPLES + BATCH_SIZE
        correct = sess.run(correct_prediction_sum, feed_dict={batch_size: BATCH_SIZE, XTEST: x_test[start:end], YTEST: y_test[start:end]})

        count += BATCH_SIZE
        total_correct += correct

    print (total_correct * 1.0 / count)
    sys.stdout.flush()
    
    f = open(filename, "a")
    f.write(str(total_correct * 1.0 / count) + "\n")
    f.close()

##############################################

f.close()

