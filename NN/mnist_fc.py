
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--name', type=str, default=None)
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--shuffle', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import time
import tensorflow as tf
import keras
from keras.datasets import mnist
import math
import numpy as np

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution
from MaxPool import MaxPool
from Dropout import Dropout
from FeedbackFC import FeedbackFC

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu
from Activation import Linear

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
sparse = args.sparse

##############################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

if args.shuffle:
    print ("Shuffling!")
    perm = np.random.permutation(TRAIN_EXAMPLES)

    tmp1 = np.copy(x_train[0])
    x_train[perm] = x_train
    y_train[perm] = y_train
    tmp2 = x_train[perm[0]]
    
    assert(np.all(tmp1 == tmp2))
    
x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

##############################################

#tf.set_random_seed(0)
#tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
XTRAIN = tf.placeholder(tf.float32, [None, 784])
YTRAIN = tf.placeholder(tf.float32, [None, 10])
#XTRAIN = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), XTRAIN)

XTEST = tf.placeholder(tf.float32, [None, 784])
YTEST = tf.placeholder(tf.float32, [None, 10])
#XTEST = tf.map_fn(lambda frame1: tf.image.per_image_standardization(frame1), XTEST)

l0 = FullyConnected(size=[784, 100], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Tanh(), last_layer=False)
l1 = FeedbackFC(size=[784, 100], num_classes=10, sparse=sparse, rank=args.rank, load=args.load)

l2 = FullyConnected(size=[100, 10], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Sigmoid(), last_layer=True)

model = Model(layers=[l0, l1, l2])

##############################################

predict = model.predict(X=XTEST)

W1 = l0.get_weights()
B = l1.get_feedback()
W2 = l2.get_weights()

if args.dfa:
    train = model.dfa(X=XTRAIN, Y=YTRAIN)
else:
    train = model.train(X=XTRAIN, Y=YTRAIN)
    
correct = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

##############################################

filename = "mnist_" +                   \
           str(args.epochs) + "_" +     \
           str(args.batch_size) + "_" + \
           str(args.alpha) + "_" +      \
           str(args.dfa) + "_" +        \
           str(args.sparse) + "_" +     \
           str(args.gpu) + "_" +        \
           args.init + "_" +            \
           args.opt +                   \
           ".results"

f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

accs = []

for ii in range(EPOCHS):
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        sess.run([train], feed_dict={batch_size: BATCH_SIZE, XTRAIN: xs, YTRAIN: ys})
        
    total_correct_examples = 0.0
    total_examples = 0.0

    for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        tmp = sess.run(total_correct, feed_dict={batch_size: BATCH_SIZE, XTEST: xs, YTEST: ys})
        total_correct_examples += tmp
        total_examples += BATCH_SIZE
            
    print ("acc: " + str(total_correct_examples / total_examples))
    accs.append(total_correct_examples / total_examples)    
    
    f = open(filename, "a")
    f.write(str(total_correct_examples * 1.0 / total_examples) + "\n")
    f.close()

##############################################

if args.name is not None:
    np.save(args.name, np.array(accs))
    
if args.save:
    w1, b, w2 = sess.run([W1, B, W2])
    
    w1_name = "W1_" + str(args.num) + "_" + str(args.gpu)
    np.save(w1_name, w1)
    
    w2_name = "W2_" + str(args.num) + "_" + str(args.gpu)
    np.save(w2_name, w2)
    
    acc_name = "acc_" + str(args.num) + "_" + str(args.gpu)
    np.save(acc_name, np.array(accs))   
    
    B_name = "B_" + str(args.num) + "_" + str(args.gpu)
    np.save(B_name, b)




