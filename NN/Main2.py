
import os

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=str(0)

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
from Dropout import Dropout

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 20
ALPHA = 1e-3

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 32]) * (2 * 0.12) - 0.12)
l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

W1 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 64]) * (2 * 0.12) - 0.12)
l1 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], filters=W1, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l2 = MaxPool(size=[batch_size, 28, 28, 64], stride=[1, 2, 2, 1])

l3 = Dropout(rate=0.25)

l4 = ConvToFullyConnected(shape=[14, 14, 64])

W5 = tf.Variable(tf.random_uniform(shape=[14*14*64, 128]) * (2 * 0.12) - 0.12)
l5 = FullyConnected(size=[14*14*64, 128], weights=W5, alpha=ALPHA, activation=Relu(), last_layer=False)

l6 = Dropout(rate=0.5)

W7 = tf.Variable(tf.random_uniform(shape=[128, 10]) * (2 * 0.12) - 0.12)
l7 = FullyConnected(size=[128, 10], weights=W7, alpha=ALPHA, activation=Relu(), last_layer=False)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7])

##############################################

ret = model.train(X=X, Y=Y)
predict = model.predict(X=X)

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
total_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))

acc, acc_op = tf.metrics.accuracy(labels=tf.argmax(Y, 1), predictions=tf.argmax(predict, 1))

##############################################

sess = tf.InteractiveSession()
tf.local_variables_initializer().run()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(EPOCHS):
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        print (str(ii * TRAIN_EXAMPLES + jj * BATCH_SIZE) + "/" + str(int(EPOCHS * TRAIN_EXAMPLES)))
        batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
        batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
        # train
        sess.run(ret, feed_dict={batch_size: BATCH_SIZE, X: batch_xs, Y: batch_ys})
end = time.time()

correct = 0
total = 0
for ii in range(int(TEST_EXAMPLES / BATCH_SIZE)):
    print (str(ii * BATCH_SIZE) + "/" + str(int(TEST_EXAMPLES)))
    batch_xs, batch_ys = mnist.test.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    
    # test
    batch_correct_count = sess.run(total_correct, feed_dict={batch_size: BATCH_SIZE, X: batch_xs, Y: batch_ys})
    correct += batch_correct_count
    total += BATCH_SIZE
    # test
    print('acc:', sess.run(acc_op, feed_dict={batch_size: BATCH_SIZE, X: batch_xs, Y: batch_ys}))


##############################################
total_accuracy = correct / total
print ("correct: " + str(correct) + " total: " + str(total) + " accuracy: " + str(total_accuracy))
##############################################
print("time taken: " + str(end - start))








