
import argparse
import os

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, nargs='*')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

num_layers = len(args.layers)
assert(num_layers >= 2)

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import tensorflow as tf
import numpy as np
import keras
from nn_tf import nn_tf

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size
ALPHA = args.alpha

##############################################

W = [None] * (num_layers-1)
for ii in range(0, num_layers-1):
    W[ii] = tf.Variable(tf.random_uniform(shape=[args.layers[ii]+1, args.layers[ii+1]]) * (2 * 0.12) - 0.12)

X = tf.placeholder(tf.float32, [None, args.layers[0]])
Y = tf.placeholder(tf.float32, [None, args.layers[-1]])

model = nn_tf(size=args.layers,  \
              weights=W,         \
              alpha=ALPHA,       \
              bias=True)
# predict
predict = model.predict(X)

# train     
ret = model.train(X, Y)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

x_train = x_train.reshape(TRAIN_EXAMPLES, args.layers[0])
x_train = x_train / 255.
y_train = keras.utils.to_categorical(y_train, args.layers[-1])

x_test = x_test.reshape(TEST_EXAMPLES, args.layers[0])
x_test = x_test / 255.
y_test = keras.utils.to_categorical(y_test, args.layers[-1])

for ii in range(0, EPOCHS * TRAIN_EXAMPLES, BATCH_SIZE):
    start = ii % TRAIN_EXAMPLES
    end = ii % TRAIN_EXAMPLES + BATCH_SIZE
    sess.run([ret], feed_dict={X: x_train[start:end], Y: y_train[start:end]})

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))












