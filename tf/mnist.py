
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# specify which GPU(s) to be use
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def add_bias(x):
  return tf.concat([x, tf.ones([tf.shape(x)[0], 1])], axis=1)

##############################################
W1 = tf.Variable(tf.random_uniform(shape=[785, 100]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[101, 10]) * (2 * 0.12) - 0.12)
##############################################
# FEED FORWARD
##############################################
X = tf.placeholder(tf.float32, [None, 784])
A1 = add_bias(X)

Y2 = tf.matmul(A1, W1)
A2 = add_bias(tf.sigmoid(Y2))

Y3 = tf.matmul(A2, W2)
A3 = tf.sigmoid(Y3)
##############################################
# BACK PROP
##############################################
ANS = tf.placeholder(tf.float32, [None, 10])
D3 = tf.subtract(A3, ANS)
D2 = tf.multiply(tf.matmul(D3, tf.transpose(W2)), tf.multiply(A2, tf.subtract(1.0, A2)))

G2 = tf.matmul(tf.transpose(A2), D3)
G1 = tf.matmul(tf.transpose(A1), D2[:, :-1])

W2 = W2.assign(tf.subtract(W2, tf.scalar_mul(1e-2, G2)))
W1 = W1.assign(tf.subtract(W1, tf.scalar_mul(1e-2, G1)))
##############################################

# SPECIFY HOW MUCH MEMORY TO USE
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

TRAIN_EXAMPLES = 50000
EPOCHS = 10
BATCH_SIZE = 32

for ii in range(int(TRAIN_EXAMPLES * EPOCHS / BATCH_SIZE)):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run([W1, W2], feed_dict={X: batch_xs, ANS: batch_ys})

correct_prediction = tf.equal(tf.argmax(A3,1), tf.argmax(ANS,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, ANS: mnist.test.labels}))

##############################################












