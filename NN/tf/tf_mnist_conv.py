from __future__ import print_function

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = args.epochs
TRAINING_EXAMPLES = 60000
TESTING_EXAMPLES = 10000
learning_rate = args.alpha

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAINING_EXAMPLES, 28, 28, 1)
x_test = x_test.reshape(TESTING_EXAMPLES, 28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

####################################

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

conv1 = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)

conv2_pool = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2, padding='same')

flat = tf.contrib.layers.flatten(conv2_pool)

fc1 = tf.layers.dense(inputs=flat, units=128, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, units=10)

predict = tf.argmax(fc2, axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc2, labels=y))
correct = tf.equal(predict, tf.argmax(y, 1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1).minimize(loss)

####################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(EPOCHS):
    for jj in range(int(TRAINING_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        sess.run([optimizer], feed_dict ={x: xs, y: ys})
        
    total_correct_examples = 0.0
    total_examples = 0.0

    for jj in range(int(TESTING_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        tmp = sess.run(total_correct, feed_dict ={x: xs, y: ys})
        total_correct_examples += tmp
        total_examples += BATCH_SIZE
            
    print ("acc: " + str(total_correct_examples / total_examples))
        
        
