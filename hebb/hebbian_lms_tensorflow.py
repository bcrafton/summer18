
import os
import sys
import tensorflow as tf
import time

import numpy as np
import math
import cPickle as pickle
import gzip

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)

  f.close()

dtype = tf.float32
load_data()

##########################
# with tf.device("/cpu:0"):
with tf.device("/gpu:0"):
  w = tf.placeholder(shape=(28*28, 20*20), dtype=dtype)
  x = tf.placeholder(shape=(28*28, 1),     dtype=dtype)
  xt = tf.transpose(x)
  xw = tf.matmul(xt, w)
  avg = tf.reduce_mean(xw) # no axis means avg of whole matrix
  xw = tf.divide(xw, avg)
  xw = tf.pow(xw, 3)
  sig = tf.sigmoid(xw)
  e = tf.subtract(sig, tf.multiply(xw, 0.5))
  gradient = tf.add(tf.multiply(tf.matmul(x, e), 0.001), w)
##########################
config = tf.ConfigProto(graph_options=tf.GraphOptions(optimizer_options=tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L0)))
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
##########################

start = time.time()

# weights = tf.random_normal(shape=(28*28, 20*20), mean=0.5, stddev=0.1)
weights = np.absolute(np.random.normal(0.5, 0.1, size=(28*28, 20*20)))
for i in range(50000):
  # move this into tensorflow.
  input = np.array(training_set[i]).reshape(28*28, 1)
  input = input / np.average(input)
  # input = tf.convert_to_tensor(input)
  
  grad = sess.run(gradient, feed_dict={x: input, w: weights})
  # print(type(grad))
  
  # move this into tensorflow.
  weights = weights + grad
  
  # move this into tensorflow.
  col_norm = np.average(weights, axis = 0)
  col_norm = 0.5 / col_norm
  for j in range(20*20):
    weights[:, j] *= col_norm[j]
  
end = time.time()
print end-start

##########################



