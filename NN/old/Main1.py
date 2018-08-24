
import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = 1
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 32
ALPHA = 1e-2

##############################################

batch_size = tf.placeholder(tf.int32)

XTRAIN = tf.placeholder(tf.float32, [None, 28, 28, 1])
YTRAIN = tf.placeholder(tf.float32, [None, 10])

XTEST = tf.placeholder(tf.float32, [None, 28, 28, 1])
YTEST = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 32]) * (2 * 0.12) - 0.12)
l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], num_classes=10, filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

W1 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 64]) * (2 * 0.12) - 0.12)
l1 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], num_classes=10, filters=W1, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l2 = ConvToFullyConnected(shape=[28, 28, 64])

W3 = tf.Variable(tf.random_uniform(shape=[28*28*64, 128]) * (2 * 0.12) - 0.12)
l3 = FullyConnected(size=[28*28*64, 128], num_classes=10, weights=W3, alpha=ALPHA, activation=Relu(), last_layer=False)

W4 = tf.Variable(tf.random_uniform(shape=[128, 10]) * (2 * 0.12) - 0.12)
l4 = FullyConnected(size=[128, 10], num_classes=10, weights=W4, alpha=ALPHA, activation=Relu(), last_layer=True)

model = Model(layers=[l0, l1, l2, l3, l4])

##############################################

predict = model.predict(X=XTEST)

ret = model.train(X=XTRAIN, Y=YTRAIN)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(int(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE)):
    print (str(ii * BATCH_SIZE) + "/" + str(int(EPOCHS * TRAIN_EXAMPLES)))
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    sess.run(ret, feed_dict={batch_size: BATCH_SIZE, XTRAIN: batch_xs, YTRAIN: batch_ys})

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(YTEST, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={batch_size: TEST_EXAMPLES, XTEST: mnist.test.images.reshape(TEST_EXAMPLES, 28, 28, 1), YTEST: mnist.test.labels}))

##############################################


