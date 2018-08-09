
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from nn_tf import nn_tf

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

LAYER1 = 784
LAYER2 = 500
LAYER3 = 250
LAYER4 = 100
LAYER5 = 10

EPOCHS = 100
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 100

##############################################

W1 = tf.Variable(tf.random_uniform(shape=[LAYER1+1, LAYER2]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[LAYER2+1, LAYER3]) * (2 * 0.12) - 0.12)
W3 = tf.Variable(tf.random_uniform(shape=[LAYER3+1, LAYER4]) * (2 * 0.12) - 0.12)
W4 = tf.Variable(tf.random_uniform(shape=[LAYER4+1, LAYER5]) * (2 * 0.12) - 0.12)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

model = nn_tf(size=[LAYER1, LAYER2, LAYER3, LAYER4, LAYER5], \
              weights=[W1, W2, W3, W4],                                      \
              alpha=1e-3,                                                    \
              bias=True)

# predict
predict = model.predict(X)

# train     
[W1, W2, W3, W4] = model.train(X, Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for ii in range(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
    sess.run([W1,W2], feed_dict={X: batch_xs, Y: batch_ys})
    print (ii * BATCH_SIZE)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))














