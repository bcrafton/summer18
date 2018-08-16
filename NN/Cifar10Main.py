
import os 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import time
import tensorflow as tf
import keras
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

from Model import Model
from Layer import Layer 
from FullyConnected import FullyConnected
from Activation import Activation
from Activation import Sigmoid

##############################################

# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = 10
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 32
ALPHA = 1e-2

##############################################

W1 = tf.Variable(tf.random_uniform(shape=[3072, 1024]) * (2 * 0.12) - 0.12)
l1 = FullyConnected(size=[3072, 1024], weights=W1, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

W2 = tf.Variable(tf.random_uniform(shape=[1024, 128]) * (2 * 0.12) - 0.12)
l2 = FullyConnected(size=[1024, 128], weights=W2, alpha=ALPHA, activation=Sigmoid(), last_layer=False)

W3 = tf.Variable(tf.random_uniform(shape=[128, 10]) * (2 * 0.12) - 0.12)
l3 = FullyConnected(size=[128, 10], weights=W3, alpha=ALPHA, activation=Sigmoid(), last_layer=True)

X = tf.placeholder(tf.float32, [None, 3072])
Y = tf.placeholder(tf.float32, [None, 10])

model = Model(layers=[l1, l2, l3])

predict = model.predict(X=X)

ret = model.train(X=X, Y=Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAIN_EXAMPLES, 3072)
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, 3072)
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, 10)

start = time.time()
for ii in range(0, EPOCHS * TRAIN_EXAMPLES, BATCH_SIZE):
    start = ii % TRAIN_EXAMPLES
    end = ii % TRAIN_EXAMPLES + BATCH_SIZE
    sess.run([ret], feed_dict={X: x_train[start:end], Y: y_train[start:end]})

end = time.time()

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
print("time taken: " + str(end - start))

##############################################





