
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from nn_tf import nn_tf

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

EPOCHS = 100
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 100

##############################################

W1 = tf.Variable(tf.random_uniform(shape=[LAYER1+1, LAYER2]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[LAYER2+1, LAYER3]) * (2 * 0.12) - 0.12)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

model = nn_tf(size=[LAYER1, LAYER2, LAYER3],          \
              weights=[W1, W2],                       \
              alpha=1e-2,                             \
              bias=True)

# predict
predict = model.predict(X)

# train     
[W1, W2] = model.train(X, Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(100000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run([W1,W2], feed_dict={X: batch_xs, Y: batch_ys})

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))














