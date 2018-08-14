
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
W = tf.Variable(tf.random_uniform(shape=[5, 5, 1, 16]) * (2 * 0.12) - 0.12)
ret = tf.nn.conv2d(X, W, [1,1,1,1], "SAME")

WT = tf.reshape(W, (5, 5, 16, 1))
# W1 = tf.Variable(tf.random_uniform(shape=[5, 5, 16, 1]) * (2 * 0.12) - 0.12)
ret1 = tf.nn.conv2d(ret, WT, [1,1,1,1], "SAME")

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    sess.run([ret, ret1], feed_dict={X: batch_xs})
    print (ret.get_shape(), ret1.get_shape())
end = time.time()

##############################################


