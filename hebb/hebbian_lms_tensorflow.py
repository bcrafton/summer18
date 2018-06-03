
# https://stackoverflow.com/questions/41804380/testing-gpu-with-tensorflow-matrix-multiplication

# On Titan X (Pascal)
# 8192 x 8192 matmul took: 0.10 sec, 11304.59 G ops/sec
# http://stackoverflow.com/questions/41804380/testing-gpu-with-tensorflow-matrix-multiplication

import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import tensorflow as tf
import time

n = 8192
dtype = tf.float32

w = tf.Variable(tf.random_uniform((28*28, 20*20), dtype=dtype))
x = tf.Variable(tf.random_uniform((28*28, 1), dtype=dtype))
xt = tf.transpose(x)
xw = tf.matmul(xt, w)
sig = tf.sigmoid(xw)
e = tf.subtract(sig, tf.multiply(xw, 0.5))
gradient = tf.add(tf.multiply(tf.matmul(x, e), 0.0001), w)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

start = time.time()
for i in range(50000):
  sess.run(gradient.op)
end = time.time()
