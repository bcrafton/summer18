
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

W1 = tf.Variable(tf.random_uniform(shape=[785, 100]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[101, 10]) * (2 * 0.12) - 0.12)
print(W2)
print(W1)
##############################################

X = tf.placeholder(tf.float32, [None, 784])
A1 = tf.concat([X, tf.ones([tf.shape(X)[0], 1])], axis=1)

Y2 = tf.matmul(A1, W1)
A2 = tf.concat([tf.sigmoid(Y2), tf.ones([tf.shape(X)[0], 1])], axis=1)

Y3 = tf.matmul(A2, W2)
A3 = tf.sigmoid(Y3)

##############################################

ANS = tf.placeholder(tf.float32, [None, 10])
D3 = tf.subtract(A3, ANS)
D2 = tf.multiply(tf.matmul(D3, tf.transpose(W2)), tf.multiply(A2, tf.subtract(1.0, A2)))

G2 = tf.matmul(tf.transpose(A2), D3)
G1 = tf.matmul(tf.transpose(A1), D2[:, :-1])

#TODO Just had to assign variable using tensorflow notation

W2 = W2.assign(tf.subtract(W2, tf.scalar_mul(1e-4, G2)))
W1 = W1.assign(tf.subtract(W1, tf.scalar_mul(1e-4, G1)))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run([W1,W2], feed_dict={X: batch_xs, ANS: batch_ys})
    print(ii)

correct_prediction = tf.equal(tf.argmax(A3,1), tf.argmax(ANS,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, ANS: mnist.test.labels}))














