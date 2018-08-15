
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

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = 100
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 32
ALPHA = 1e-2

##############################################

batch = tf.placeholder(tf.int32)

XTRAIN = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])
YTRAIN = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

XTEST = tf.placeholder(tf.float32, [TEST_EXAMPLES, 28, 28, 1])
YTEST = tf.placeholder(tf.float32, [TEST_EXAMPLES, 10])

W0 = tf.Variable(tf.random_uniform(shape=[5, 5, 1, 16]) * (2 * 0.12) - 0.12)
l0 = Convolution(input_sizes=[batch, 28, 28, 1], filter_sizes=[5, 5, 1, 16], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

l1 = ConvToFullyConnected(shape=[28, 28, 16])

W2 = tf.Variable(tf.random_uniform(shape=[28*28*16, 10]) * (2 * 0.12) - 0.12)
l2 = FullyConnected(size=[28*28*16, 10], weights=W2, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

model = Model(layers=[l0, l1, l2])

'''
XTRAIN = tf.placeholder(tf.float32, [BATCH_SIZE, 784])
YTRAIN = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

XTEST = tf.placeholder(tf.float32, [TEST_EXAMPLES, 784])
YTEST = tf.placeholder(tf.float32, [TEST_EXAMPLES, 10])

W0 = tf.Variable(tf.random_uniform(shape=[784, 100]) * (2 * 0.12) - 0.12)
l0 = FullyConnected(size=[784, 100], weights=W0, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

W1 = tf.Variable(tf.random_uniform(shape=[100, 10]) * (2 * 0.12) - 0.12)
l1 = FullyConnected(size=[100, 10], weights=W1, alpha=ALPHA, activation=Sigmoid(), last_layer=True)

model = Model(layers=[l0, l1])
'''

##############################################

ret = model.train(batch_size=BATCH_SIZE, X=XTRAIN, Y=YTRAIN)
predict = model.predict(batch_size=TEST_EXAMPLES, X=XTEST)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(int(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    sess.run(ret, feed_dict={batch: BATCH_SIZE, XTRAIN: batch_xs, YTRAIN: batch_ys})
end = time.time()

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={batch: TEST_EXAMPLES, XTRAIN: batch_xs, YTRAIN: batch_ys, XTEST: mnist.test.images.reshape(TEST_EXAMPLES, 28, 28, 1), YTEST: mnist.test.labels}))
# print(sess.run(accuracy, feed_dict={XTRAIN: batch_xs, YTRAIN: batch_ys, XTEST: mnist.test.images, YTEST: mnist.test.labels}))
print("time taken: " + str(end - start))

##############################################


