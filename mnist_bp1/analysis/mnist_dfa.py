
import os
import argparse 

parser = argparse.ArgumentParser()
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

import tensorflow as tf
import numpy as np
import keras
from keras.datasets import mnist
import matplotlib.pyplot as plt

##############################################

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

##############################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10
EPOCHS = 25
BATCH_SIZE = 1

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
  
def relu(x):
    return tf.nn.relu(x)

def drelu(x):
    return tf.cast(x > 0.0, dtype=tf.float32)
    
def sigmoid(x):
    return tf.sigmoid(x)

def dsigmoid(x):
    return tf.multiply(x, tf.subtract(1.0, x))

##############################################
high = 1.0 / np.sqrt(785)
low = -high
w1_init = np.random.uniform(low=low, high=high, size=(785, 100))
W1 = tf.Variable( tf.cast(w1_init, tf.float32) )

high = 1.0 / np.sqrt(101)
low = -high
w2_init = np.random.uniform(low=low, high=high, size=(101, 10))
W2 = tf.Variable( tf.cast(w2_init, tf.float32) )

high = 1.0 / np.sqrt(101)
low = -high
b_init = np.load(args.load)
# b_init = np.random.uniform(low=low, high=high, size=(101, 10))
# b_init = np.copy(w2_init)
B = tf.Variable(tf.cast(np.copy(b_init), tf.float32))
##############################################
# FEED FORWARD
##############################################
ALPHA = tf.placeholder(tf.float32, shape=())
X = tf.placeholder(tf.float32, [None, 784])
A1 = add_bias(X)

Y2 = tf.matmul(A1, W1)
A2 = add_bias(sigmoid(Y2))

Y3 = tf.matmul(A2, W2)
A3 = sigmoid(Y3)
##############################################
# BACK PROP
##############################################
ANS = tf.placeholder(tf.float32, [None, 10])
# D3 = tf.multiply(tf.subtract(A3, ANS), dsigmoid(A3))
D3 = tf.subtract(A3, ANS)
D2 = tf.multiply(tf.matmul(D3, tf.transpose(B)), dsigmoid(A2))

G2 = tf.matmul(tf.transpose(A2), D3)
G1 = tf.matmul(tf.transpose(A1), D2[:, :-1])

W2 = W2.assign(tf.subtract(W2, tf.scalar_mul(ALPHA, G2)))
W1 = W1.assign(tf.subtract(W1, tf.scalar_mul(ALPHA, G1)))

loss = tf.reduce_sum(tf.abs(D3))
correct_prediction = tf.equal(tf.argmax(A3,1), tf.argmax(ANS,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

b = np.copy(b_init)
b = b[0:100]
b = np.reshape(b, (-1))

angles = []
losses = []
accs = []

for ii in range(EPOCHS):
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        w1, w2, l = sess.run([W1, W2, loss], feed_dict={ALPHA: 0.01, X: xs, ANS: ys})
        
        ##############################################   
        '''  
        w2 = w2[0:100]
        w2 = np.reshape(w2, (-1))
        
        print( angle_between(w2, b) * (180.0 / 3.14) )
        print( l ) 
        
        losses.append(l)
        angles.append(angle_between(w2, b) * (180.0 / 3.14))
        '''
        ##############################################
        
    total_correct_examples = 0.0
    total_examples = 0.0

    acc, w1, w2 = sess.run([accuracy, W1, W2], feed_dict={ALPHA: 0.00, X: x_test, ANS: y_test})
    print ("acc: " + str(acc))
    
    accs.append(acc)
    
    ##############################################

'''
plt.subplot(311)
plt.plot(angles)
plt.xlabel("Angle")

plt.subplot(312)
plt.plot(losses)
plt.xlabel("Loss")

plt.subplot(313)
plt.plot(accs)
plt.xlabel("Accuracy")

plt.show()
'''

np.save("W1_" + str(args.num) + "_" + str(args.gpu), w1)
np.save("W2_" + str(args.num) + "_" + str(args.gpu), w2)

# print ("accuracy: " + str(acc))











