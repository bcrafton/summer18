
import numpy as np
import math
import gzip
import time
import pickle
import argparse
import keras
from keras.datasets import mnist

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10
EPOCHS = 1
BATCH_SIZE = 32

ALPHA = 1e-2

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

EPSILON = 0.12
 
#######################################

def relu(x):
    return (x > 0.0) * x
    
def drelu(x):
    return (x > 0.0)

#######################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)

#######################################

weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1, LAYER2)) * 2 * EPSILON - EPSILON
weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2, LAYER3)) * 2 * EPSILON - EPSILON

B = np.random.uniform(0.0, 1.0, size=(LAYER2, LAYER3)) * 2 * EPSILON - EPSILON

#######################################

for epoch in range(EPOCHS):

    print ("epoch: " + str(epoch + 1) + "/" + str(EPOCHS))
    for ex in range(TRAIN_EXAMPLES):
        A1 = x_train[ex]
        Z2 = np.dot(A1, weights1)
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2)
        A3 = relu(Z3)
        
        ANS = y_train[ex]
        
        D3 = A3 - ANS
        D2 = np.dot(D3, np.transpose(B)) * drelu(A2)

        DW2 = np.dot(A2.reshape(LAYER2, 1), D3.reshape(1, LAYER3))        
        DW1 = np.dot(A1.reshape(LAYER1, 1), D2.reshape(1, LAYER2))
        
        weights2 -= ALPHA * DW2
        weights1 -= ALPHA * DW1
        
correct = 0
for ex in range(TEST_EXAMPLES):
    A1 = x_test[ex]
    Z2 = np.dot(A1, weights1)
    A2 = relu(Z2)
    Z3 = np.dot(A2, weights2)
    A3 = relu(Z3)  
    
    if (np.argmax(A3) == np.argmax(y_test[ex])):
        correct += 1
    
print ("accuracy: " + str(1.0 * correct / TEST_EXAMPLES))
    
    
    
    
    
    
