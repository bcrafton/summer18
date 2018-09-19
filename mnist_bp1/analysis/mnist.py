
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist


##############################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10
EPOCHS = 250
BATCH_SIZE = 32

##############################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

##############################################

def add_bias(x):
  return tf.concat([x, tf.ones([tf.shape(x)[0], 1])], axis=1)

##############################################
W1 = tf.Variable(tf.random_uniform(shape=[785, 25]) * (2 * 0.12) - 0.12)
W2 = tf.Variable(tf.random_uniform(shape=[26, 10]) * (2 * 0.12) - 0.12)
##############################################
# FEED FORWARD
##############################################
ALPHA = tf.placeholder(tf.float32, shape=())
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

W2 = W2.assign(tf.subtract(W2, tf.scalar_mul(ALPHA, G2)))
W1 = W1.assign(tf.subtract(W1, tf.scalar_mul(ALPHA, G1)))

correct_prediction = tf.equal(tf.argmax(A3,1), tf.argmax(ANS,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

prev1 = np.zeros(shape=(785, 25))
prev2 = np.zeros(shape=(26, 10))

for ii in range(EPOCHS):
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        sess.run([W1, W2], feed_dict={ALPHA: 0.01, X: xs, ANS: ys})
        
    total_correct_examples = 0.0
    total_examples = 0.0

    acc, w1, w2 = sess.run([accuracy, W1, W2], feed_dict={ALPHA: 0.00, X: x_test, ANS: y_test})
    
    print (np.sum(np.absolute(prev1 - w1)))
    print (np.sum(np.absolute(prev2 - w2)))
    prev1 = w1
    prev2 = w2
    
    print ("acc: " + str(acc))

np.save("W1_" + str(args.num) + "_" + str(args.gpu), w1)
np.save("W2_" + str(args.num) + "_" + str(args.gpu), w2)

print ("accuracy: " + str(acc))












