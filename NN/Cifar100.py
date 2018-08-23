
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=5e-5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=1)
parser.add_argument('--sparse', type=int, default=1)
parser.add_argument('--init', type=str, default="zero")
parser.add_argument('--opt', type=str, default="adam")
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
from Feedback import Feedback

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu

##############################################

cifar100 = tf.keras.datasets.cifar100.load_data()

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
YTRAIN = tf.placeholder(tf.float32, [None, 100])
XTRAIN = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), XTRAIN)

XTEST = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTEST = tf.placeholder(tf.float32, [None, 100])
XTEST = tf.map_fn(lambda frame1: tf.image.per_image_standardization(frame1), XTEST)

l0 = Convolution(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], num_classes=100, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

l1 = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

l2 = Feedback(size=[batch_size, 16, 16, 96], num_classes=100, sparse=False)

l3 = Convolution(input_sizes=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], num_classes=100, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

l4 = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

l5 = Feedback(size=[batch_size, 8, 8, 128], num_classes=100, sparse=False)

l6 = Convolution(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 256], num_classes=100, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

l7 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

l8 = Feedback(size=[batch_size, 4, 4, 256], num_classes=100, sparse=False)

l9 = ConvToFullyConnected(shape=[4, 4, 256])

l10 = FullyConnected(size=[4*4*256, 2048], num_classes=100, init_weights=args.init, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

l11 = FullyConnected(size=[2048, 2048], num_classes=100, init_weights=args.init, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

l12 = FullyConnected(size=[2048, 100], num_classes=100, init_weights=args.init, alpha=ALPHA, activation=Tanh(), last_layer=True, sparse=sparse)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12])

predict = model.predict(X=XTEST)

if args.dfa:
    grads_and_vars = model.dfa(X=XTRAIN, Y=YTRAIN)
else:
    grads_and_vars = model.train(X=XTRAIN, Y=YTRAIN)
    
if args.opt == "adam":
    optimizer = tf.train.AdamOptimizer(learning_rate=ALPHA, beta1=0.9, beta2=0.999, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)
elif args.opt == "rms":
    optimizer = tf.train.RMSPropOptimizer(learning_rate=ALPHA, decay=1.0, momentum=0.0).apply_gradients(grads_and_vars=grads_and_vars)
else:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=ALPHA).apply_gradients(grads_and_vars=grads_and_vars)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
correct_prediction_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar100

x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
y_train = keras.utils.to_categorical(y_train, 100)

x_test = x_test.reshape(TEST_EXAMPLES, 32, 32, 3)
y_test = keras.utils.to_categorical(y_test, 100)

##############################################

filename = "cifaf100" + str(args.epochs) + "_" \
                      + str(args.batch_size) + "_" \
                      + str(args.alpha) + "_" \
                      + str(args.dfa) + "_" \
                      + str(args.sparse) + "_" \
                      + str(args.gpu) + "_" \
                      + args.init + "_" \
                      + args.opt + ".results"

f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
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

