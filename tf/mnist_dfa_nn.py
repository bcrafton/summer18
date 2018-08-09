
import argparse
import os

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--layers', type=int, nargs='*')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=int, default=1e-2)
parser.add_argument('--gpu', type=int, default=-1)
args = parser.parse_args()

num_layers = len(args.layers)
assert(num_layers >= 2)

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from nn_dfa_tf import nn_dfa_tf

##############################################

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size
ALPHA = args.alpha

##############################################

# we need tf random seed lmao
# np.random.seed(0)

'''
b = [None] * (num_layers-1)
for ii in range(0, num_layers-1):
    b[ii] = np.zeros(shape=(args.layers[ii], args.layers[num_layers-1]))
    for jj in range(args.layers[ii]):
        idx = int(np.random.randint(0, args.layers[num_layers-1]))
        b[ii][jj][idx] = np.random.uniform(0.25, 0.75)
'''

W = [None] * (num_layers-1)
B = [None] * (num_layers-1)
for ii in range(0, num_layers-1):
    W[ii] = tf.Variable(tf.random_uniform(shape=[args.layers[ii]+1, args.layers[ii+1]]) * (2 * 0.12) - 0.12)
    
    # B[ii] = tf.get_variable("B"+str(ii), initializer=b[ii])
    
    b = np.zeros(shape=(args.layers[ii], args.layers[num_layers-1]))
    for jj in range(args.layers[ii]):
        idx = int(np.random.randint(0, args.layers[num_layers-1]))
        b[jj][idx] = np.random.uniform(0.25, 0.75)
    B[ii] = tf.cast(tf.Variable(b), tf.float32)
    
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

model = nn_dfa_tf(size=args.layers,  \
                  weights=W,         \
                  B=B,               \
                  alpha=ALPHA,       \
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
end = time.time()

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels}))
print("time taken: " + str(end - start))

##############################################


