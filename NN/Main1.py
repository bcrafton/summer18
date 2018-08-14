
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

W0 = tf.Variable(tf.random_uniform(shape=[5, 5, 1, 16]) * (2 * 0.12) - 0.12)
l0 = Convolution(input_sizes=[BATCH_SIZE, 28, 28, 1], filter_sizes=[5, 5, 1, 16], filters=W0, stride=1, padding=1, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

l1 = ConvToFullyConnected(shape=[28, 28, 16])

W2 = tf.Variable(tf.random_uniform(shape=[28*28*16, 10]) * (2 * 0.12) - 0.12)
l2 = FullyConnected(size=[28*28*16, 10], weights=W2, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

X = tf.placeholder(tf.float32, [BATCH_SIZE, 28, 28, 1])
Y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])

model = Model(layers=[l0, l1, l2])
ret = model.train(X=X, Y=Y)
predict = model.predict(X=X)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(int(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    batch_xs = batch_xs.reshape(BATCH_SIZE, 28, 28, 1)
    sess.run(ret, feed_dict={X: batch_xs, Y: batch_ys})
end = time.time()

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
print("time taken: " + str(end - start))

##############################################


