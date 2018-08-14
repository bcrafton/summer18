
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from Model import Model
from Layer import Layer 
from FullyConnected import FullyConnected
from Convolution import Convolution
from Activation import Activation
from Activation import Sigmoid

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 32
ALPHA = 1e-2

##############################################

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
W = tf.Variable(tf.random_uniform(shape=[5, 5, 1, 16]))
Y = tf.nn.conv2d(X, W, [1,1,1,1], "SAME")
DF = tf.nn.conv2d_backprop_filter(input=X, filter_sizes=[5, 5, 1, 16], out_backprop=Y, strides=[1,1,1,1], padding="SAME")
DI = tf.nn.conv2d_backprop_input(input_sizes=[BATCH_SIZE, 28, 28, 1], filter=W, out_backprop=Y, strides=[1,1,1,1], padding="SAME")

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    sess.run([DF, DI], feed_dict={X: batch_xs})
    print (DF.get_shape(), DI.get_shape())
end = time.time()

##############################################


