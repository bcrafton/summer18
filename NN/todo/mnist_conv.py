
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=5e-5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=1)
parser.add_argument('--sparse', type=int, default=1)
parser.add_argument('--init', type=str, default="zero")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--imgs', type=int, default=0)
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import scipy.misc

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
from Activation import Linear

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
sparse = args.sparse

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

l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)
l1 = Feedback(size=[batch_size, 28, 28, 32], num_classes=10, sparse=sparse)

l2 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)
l3 = MaxPool(size=[batch_size, 28, 28, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
l4 = Feedback(size=[batch_size, 14, 14, 64], num_classes=10, sparse=sparse)
l5 = Dropout(rate=0.25)

l6 = ConvToFullyConnected(shape=[14, 14, 64])
l7 = FullyConnected(size=[14*14*64, 128], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Tanh(), last_layer=False, sparse=sparse)

l8 = Dropout(rate=0.5)

l9 = FullyConnected(size=[128, 10], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Linear(), last_layer=True, sparse=sparse)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9])

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

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
correct_prediction_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

if args.imgs:
    convolved_image = model.layers[0].forward(X=XTEST)
    convolved_image1 = model.up_to(X=XTEST, N=3)

##############################################

sess = tf.InteractiveSession()
tf.local_variables_initializer().run()
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
    print (ii)
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
        batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
        
        sess.run([grads_and_vars, optimizer], feed_dict={batch_size: BATCH_SIZE, XTRAIN: batch_xs, YTRAIN: batch_ys})

    count = 0
    total_correct = 0
    
    for jj in range(0, int(TEST_EXAMPLES/BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE, shuffle=False)
        batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
        
        correct = sess.run(correct_prediction_sum, feed_dict={batch_size: BATCH_SIZE, XTEST: batch_xs, YTEST: batch_ys})
        count += BATCH_SIZE
        total_correct += correct

    print (total_correct * 1.0 / count)
    sys.stdout.flush()
    
    f = open(filename, "a")
    f.write(str(total_correct * 1.0 / count) + "\n")
    f.close()

##############################################

if args.imgs:
    batch_xs, batch_ys = mnist.test.next_batch(1, shuffle=False)
    batch_xs = batch_xs.reshape(1, 28, 28, 1)

    convolved_image, convolved_image1 = sess.run([convolved_image, convolved_image1], feed_dict={batch_size: 1, XTEST: batch_xs, YTEST: batch_ys})
    
    ###################################################################
    
    convolved_image = np.reshape(convolved_image, (28, 28, 32))
    convolved_image = np.transpose(convolved_image)
    
    '''
    for ii in range(32):
        if ii == 0:
            img = convolved_image[ii]
        else:
            img = np.concatenate((img, convolved_image[ii]), axis=1)
    '''
    
    for ii in range(8):
        for jj in range(4):
            if jj == 0:
                row = convolved_image[ii * 4 + jj]
            else:
                row = np.concatenate((row, convolved_image[ii * 4 + jj]), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
         
    plt.imsave("img.png", img)
    
    ###################################################################
      
    convolved_image1 = np.reshape(convolved_image1, (28, 28, 64))
    convolved_image1 = np.transpose(convolved_image1)
    
    for ii in range(8):
        for jj in range(8):
            if jj == 0:
                row = convolved_image1[ii * 8 + jj]
            else:
                row = np.concatenate((row, convolved_image1[ii * 8 + jj]), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
         
    plt.imsave("img1.png", img)
            
    ###################################################################

