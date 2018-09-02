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
NUM_CLASSES = 100
EPOCHS = args.epochs
TRAINING_EXAMPLES = 50000
TESTING_EXAMPLES = 10000
learning_rate = args.alpha

cifar100 = tf.keras.datasets.cifar100.load_data()

(x_train, y_train), (x_test, y_test) = cifar100

x_train = x_train.reshape(TRAINING_EXAMPLES, 3072)
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 100)

x_test = x_test.reshape(TESTING_EXAMPLES, 3072)
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, 100)

####################################

x = tf.placeholder(tf.float32, [None, 3072])
y = tf.placeholder(tf.float32, [None, 100])

l1 = tf.layers.dense(x, 1000, activation=tf.nn.relu)
l2 = tf.layers.dense(l1, 1000, activation=tf.nn.relu)
l3 = tf.layers.dense(l2, 1000, activation=tf.nn.relu)
l4 = tf.layers.dense(l3, 100)

predict = tf.argmax(l4, axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=l4, labels=y))
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
        
        
