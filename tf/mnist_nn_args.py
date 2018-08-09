
import time
import tensorflow as tf
import numpy as np
import argparse
from tensorflow.examples.tutorials.mnist import input_data
from nn_tf import nn_tf

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, nargs='*')
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

num_layers = len(args.layers)
assert(num_layers >= 2)

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = 128

##############################################

np.random.seed(0)
W = [None] * (num_layers-1)
for ii in range(0, num_layers-1):
    W[ii] = tf.Variable(tf.random_uniform(shape=[args.layers[ii]+1, args.layers[ii+1]]) * (2 * 0.12) - 0.12)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

model = nn_tf(size=args.layers,  \
              weights=W,         \
              alpha=1e-2,        \
              bias=True)

# predict
predict = model.predict(X)

# train     
ret = model.train(X, Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

start = time.time()
for ii in range(EPOCHS * TRAIN_EXAMPLES / BATCH_SIZE):
    batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE, shuffle=False)
    sess.run(ret, feed_dict={X: batch_xs, Y: batch_ys})
    print (ii * BATCH_SIZE)
end = time.time()

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
print("time taken: " + str(end - start))

##############################################


