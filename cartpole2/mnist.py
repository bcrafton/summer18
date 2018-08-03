
import numpy as np
import math
import gzip
import time
import pickle
import argparse
from nn import NN

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

ALPHA = 1e-3

LAYER1 = 784
LAYER2 = 100
LAYER3 = 25
LAYER4 = 10

EPSILON = 0.12

weights1 = np.random.uniform(0.0, 1.0, size=(LAYER1 + 1, LAYER2)) * 2 * EPSILON - EPSILON
weights2 = np.random.uniform(0.0, 1.0, size=(LAYER2 + 1, LAYER3)) * 2 * EPSILON - EPSILON
weights3 = np.random.uniform(0.0, 1.0, size=(LAYER3 + 1, LAYER4)) * 2 * EPSILON - EPSILON

NUM_TRAIN_EPOCHS = 5
NUM_TRAIN_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 10000

nn = NN(size=[LAYER1, LAYER2, LAYER3, LAYER4], weights=[weights1, weights2, weights3], alpha=ALPHA, bias=True)

#######################################

for epoch in range(NUM_TRAIN_EPOCHS):    
    print "epoch: " + str(epoch + 1) + "/" + str(NUM_TRAIN_EPOCHS)
    for ex in range(NUM_TRAIN_EXAMPLES):        
        y = one_hot(training_labels[ex], 9)
        nn.train(training_set[ex], y)
        
correct = 0
for ex in range(NUM_TEST_EXAMPLES):
    out = nn.predict(testing_set[ex])
    if (np.argmax(out) == testing_labels[ex]):
        correct += 1
    
print "accuracy: " + str(1.0 * correct / NUM_TEST_EXAMPLES)
    
    
    
    
    
    
