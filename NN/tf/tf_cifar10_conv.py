'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

import tensorflow as tf

BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 100
TRAINING_EXAMPLES = 50000
TESTING_EXAMPLES = 10000
learning_rate = 5e-3

cifar10 = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAINING_EXAMPLES, 32, 32, 3)
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TESTING_EXAMPLES, 32, 32, 3)
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, 10)

####################################

x = tf.placeholder(tf.float32, [None, 32 , 32 , 3])
y = tf.placeholder(tf.float32, [None, 10])

conv1 = tf.layers.conv2d(inputs=x, filters=96, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv1_pool = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2, padding='same')

conv2 = tf.layers.conv2d(inputs=conv1_pool, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv2_pool = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=2, padding='same')

conv3 = tf.layers.conv2d(inputs=conv2_pool, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
conv3_pool = tf.layers.max_pooling2d(inputs=conv3, pool_size=[3,3], strides=2, padding='same')

flat = tf.contrib.layers.flatten(conv3_pool)

fc1 = tf.layers.dense(inputs=flat, units=2048, activation=tf.nn.relu)
fc2 = tf.layers.dense(inputs=fc1, units=2048, activation=tf.nn.relu)
fc3 = tf.layers.dense(inputs=fc2, units=10)

predict = tf.argmax(fc3, axis=1)

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc3, labels=y))
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
        
        
