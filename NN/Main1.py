
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

EPOCHS = 500
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 1000
BATCH_SIZE = 32
ALPHA = 1e-2

##############################################

batch_size = tf.placeholder(tf.int32)
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

W0 = tf.Variable(tf.random_uniform(shape=[3, 3, 1, 32]) * (2 * 0.12) - 0.12)
l0 = Convolution(input_sizes=[batch_size, 28, 28, 1], filter_sizes=[3, 3, 1, 32], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

W1 = tf.Variable(tf.random_uniform(shape=[3, 3, 32, 64]) * (2 * 0.12) - 0.12)
l1 = Convolution(input_sizes=[batch_size, 28, 28, 32], filter_sizes=[3, 3, 32, 64], filters=W1, stride=1, padding=1, alpha=ALPHA, activation=Relu(), last_layer=False)

l2 = ConvToFullyConnected(shape=[28, 28, 64])

W3 = tf.Variable(tf.random_uniform(shape=[28*28*64, 128]) * (2 * 0.12) - 0.12)
l3 = FullyConnected(size=[28*28*64, 128], weights=W3, alpha=ALPHA, activation=Relu(), last_layer=False)

W4 = tf.Variable(tf.random_uniform(shape=[128, 10]) * (2 * 0.12) - 0.12)
l4 = FullyConnected(size=[128, 10], weights=W4, alpha=ALPHA, activation=Relu(), last_layer=False)

model = Model(layers=[l0, l1, l2, l3, l4])

##############################################

ret = model.train(X=X, Y=Y)
predict = model.predict(X=X)

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

correct_prediction = tf.equal(tf.argmax(predict, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={batch_size: TEST_EXAMPLES, X: mnist.test.images[0:TEST_EXAMPLES].reshape(TEST_EXAMPLES, 28, 28, 1), Y: mnist.test.labels[0:TEST_EXAMPLES]}))
print("time taken: " + str(end - start))

##############################################


