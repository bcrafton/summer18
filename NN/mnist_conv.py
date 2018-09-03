
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=5e-3)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import time
import tensorflow as tf
import keras
from keras.datasets import mnist
import math
import numpy as np

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution
from MaxPool import MaxPool
from Dropout import Dropout
from FeedbackFC import FeedbackFC
from FeedbackConv import FeedbackConv

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu
from Activation import Linear

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
sparse = args.sparse

##############################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 28, 28, 1)
x_test = x_test.reshape(TEST_EXAMPLES, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
XTRAIN = tf.placeholder(tf.float32, [None, 28, 28, 1])
YTRAIN = tf.placeholder(tf.float32, [None, 10])
XTRAIN = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), XTRAIN)

XTEST = tf.placeholder(tf.float32, [None, 28, 28, 1])
YTEST = tf.placeholder(tf.float32, [None, 10])
XTEST = tf.map_fn(lambda frame1: tf.image.per_image_standardization(frame1), XTEST)

l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False)
l1 = FeedbackConv(size=[batch_size, 28, 28, 32], num_classes=10, sparse=sparse, rank=args.rank)

l2 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False)
l3 = MaxPool(size=[batch_size, 28, 28, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
l4 = FeedbackConv(size=[batch_size, 14, 14, 64], num_classes=10, sparse=sparse, rank=args.rank)

l5 = ConvToFullyConnected(shape=[14, 14, 64])
l6 = FullyConnected(size=[14*14*64, 128], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Tanh(), last_layer=False)
l7 = FeedbackFC(size=[14*14*64, 128], num_classes=10, sparse=sparse, rank=args.rank)

l8 = FullyConnected(size=[128, 10], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Linear(), last_layer=True)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8])

##############################################

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

correct = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

##############################################

filename = "mnist_conv_" +              \
           str(args.epochs) + "_" +     \
           str(args.batch_size) + "_" + \
           str(args.alpha) + "_" +      \
           str(args.dfa) + "_" +        \
           str(args.sparse) + "_" +     \
           str(args.gpu) + "_" +        \
           args.init + "_" +            \
           args.opt +                   \
           ".results"

f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

for ii in range(EPOCHS):
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        sess.run([optimizer], feed_dict={batch_size: BATCH_SIZE, XTRAIN: xs, YTRAIN: ys})
        
    total_correct_examples = 0.0
    total_examples = 0.0

    for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        tmp = sess.run(total_correct, feed_dict={batch_size: BATCH_SIZE, XTEST: xs, YTEST: ys})
        total_correct_examples += tmp
        total_examples += BATCH_SIZE
            
    print ("acc: " + str(total_correct_examples / total_examples))
    
    f = open(filename, "a")
    f.write(str(total_correct_examples * 1.0 / total_examples) + "\n")
    f.close()

##############################################

