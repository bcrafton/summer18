
import time
import tensorflow as tf
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

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = 1
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 128
ALPHA = 1e-2

##############################################

batch_size = tf.placeholder(tf.int32, shape=())
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
TESTY = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 32]) * (2 * 0.12) - 0.12)
l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

W1 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 64]) * (2 * 0.12) - 0.12)
l1 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], filters=W1, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l2 = MaxPool(size=[batch_size, 28, 28, 64], stride=[1, 2, 2, 1])

l3 = ConvToFullyConnected(shape=[14, 14, 64])

W4 = tf.Variable(tf.random_uniform(shape=[14*14*64, 128]) * (2 * 0.12) - 0.12)
l4 = FullyConnected(size=[14*14*64, 128], weights=W4, alpha=ALPHA, activation=Relu(), last_layer=False)

W5 = tf.Variable(tf.random_uniform(shape=[128, 10]) * (2 * 0.12) - 0.12)
l5 = FullyConnected(size=[128, 10], weights=W5, alpha=ALPHA, activation=Relu(), last_layer=False)

model = Model(layers=[l0, l1, l2, l3, l4, l5])

##############################################

ret = model.train(X=X, Y=Y)
predict = model.predict(X=X)

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(int(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE)):
    print (str(ii * BATCH_SIZE) + "/" + str(int(EPOCHS * TRAIN_EXAMPLES)))
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    sess.run(ret, feed_dict={batch_size: BATCH_SIZE, X: batch_xs, Y: batch_ys})
end = time.time()

correct_sum = 0
for ii in range(int(TEST_EXAMPLES / BATCH_SIZE)):
    print (str(ii * BATCH_SIZE) + "/" + str(int(TEST_EXAMPLES)))
    batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    batch_correct_count = sess.run(total_correct, feed_dict={batch_size: BATCH_SIZE, X: batch_xs, Y: batch_ys})
    correct_sum += batch_correct_count

total_accuracy = correct_sum / TEST_EXAMPLES
print ("accuracy: " + str(total_accuracy))
##############################################

print("time taken: " + str(end - start))








