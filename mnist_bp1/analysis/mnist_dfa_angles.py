
import numpy as np
import math
import gzip
import time
import pickle
import argparse

#######################################

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f) 
  f.close()

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def sigmoid_gradient(x):
  gz = sigmoid(x)
  ret = gz * (1 - gz)
  return ret
    
def one_hot(value, max_value):
    # zero is first entry.
    assert(max_value == 9)
    
    ret = np.zeros(10)
    for ii in range(10):
        if value == ii:
            ret[ii] = 1
    
    return ret
 
#######################################
    
load_data()
np.random.seed(0)

#######################################

# ALPHA = 0.5 / 5e4
# ALPHA = ALPHA * 10
ALPHA = 1e-2

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

EPSILON = 0.12

# ignoring bias for now
weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1 + 1, LAYER2)) * 2 * EPSILON - EPSILON
weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON

# b2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
b2 = np.zeros(shape=(LAYER2 + 1, LAYER3))
for ii in range(LAYER2):
    idx = np.random.randint(0, 9)
    b2[ii][idx] = np.random.uniform(0.0, 1.0) * 2 * EPSILON - EPSILON

# print (b2)

NUM_TRAIN_EPOCHS = 25
NUM_TRAIN_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 10000

#######################################

for epoch in range(NUM_TRAIN_EPOCHS):

    print "epoch: " + str(epoch + 1) + "/" + str(NUM_TRAIN_EPOCHS)
    for ex in range(NUM_TRAIN_EXAMPLES):
        A1 = np.append(training_set[ex], 1)
        Z2 = np.dot(A1, weights1)
        A2 = np.append(sigmoid(Z2), 1)
        Z3 = np.dot(A2, weights2)
        A3 = sigmoid(Z3)
        
        ANS = one_hot(training_labels[ex], 9)
        
        D3 = A3 - ANS
        
        D2 = np.dot(D3, np.transpose(b2))
        # D2 = np.random.normal(loc=np.average(D2), scale=np.std(D2), size=(np.shape(D2)))
        D2 = D2 * np.append(sigmoid_gradient(Z2), 1)
        
        LOCAL_G2 = np.dot(A2.reshape(LAYER2 + 1, 1), D3.reshape(1, LAYER3))
        LOCAL_G1 = np.dot(A1.reshape(LAYER1 + 1, 1), D2[:-1].reshape(1, LAYER2))
        
        weights2 -= ALPHA * LOCAL_G2
        weights1 -= ALPHA * LOCAL_G1
        
    if ((epoch+1) % 5 == 0):
        correct = 0
        for ex in range(NUM_TEST_EXAMPLES):
            A1 = np.append(testing_set[ex], 1)
            Z2 = np.dot(A1, weights1)
            A2 = np.append(sigmoid(Z2), 1)
            Z3 = np.dot(A2, weights2)
            A3 = sigmoid(Z3)    
            
            if (np.argmax(A3) == testing_labels[ex]):
                correct += 1
            
        print "accuracy: " + str(1.0 * correct / NUM_TEST_EXAMPLES)
    
    
    
    
    
    
