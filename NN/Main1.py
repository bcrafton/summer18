
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

EPOCHS = 100
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 32
ALPHA = 1e-2

##############################################

W0 = tf.Variable(tf.random_uniform(shape=[5, 5, 3, 16]) * (2 * 0.12) - 0.12)
l0 = Convolution(size=[5, 5, 3, 16], weights=W0, stride=1, padding=1, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

W1 = tf.Variable(tf.random_uniform(shape=[784, 100]) * (2 * 0.12) - 0.12)
B1 = tf.Variable(tf.random_uniform(shape=[100, 784]) * (2 * 0.12) - 0.12)
l1 = FullyConnected(size=[784, 100], weights=W1, B=B1, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

W2 = tf.Variable(tf.random_uniform(shape=[100, 10]) * (2 * 0.12) - 0.12)
B2 = tf.Variable(tf.random_uniform(shape=[10, 100]) * (2 * 0.12) - 0.12)
l2 = FullyConnected(size=[100, 10], weights=W2, B=B2, alpha=ALPHA, activation=Sigmoid(), last_layer=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

model = Model(layers=[l1, l2])
ret = model.train(X=X, Y=Y)
predict = model.predict(X=X)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(int(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    sess.run(ret, feed_dict={X: batch_xs, Y: batch_ys})
end = time.time()

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
print("time taken: " + str(end - start))

##############################################


