
import numpy as np
import math
import numpy as np
import cPickle as pickle
import gzip
import time
import argparse
from scipy.stats import percentileofscore

from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

#####################################

layer1 = 784
layer2 = 400

#####################################

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(layer1)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(layer1)

  f.close()

load_data()

#####################################

'''
scaler = StandardScaler()
scaler.fit(training_set)

training_set = scaler.transform(training_set)
testing_set = scaler.transform(testing_set)
'''

for ii in range(50000):
    training_set[ii] = training_set[ii] / np.average(training_set[ii])
    training_set[ii] = training_set[ii] / np.max(training_set[ii])
for ii in range(10000):    
    testing_set[ii] = testing_set[ii] / np.average(testing_set[ii])
    testing_set[ii] = testing_set[ii] / np.max(testing_set[ii])

#####################################

# pca = PCA(.95)
# pca.fit(training_set)

# can see values i think: 
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# print pca.singular_values_

# comment this out bc turns it into 339 array.
# training_set = pca.transform(training_set)
# testing_set = pca.transform(testing_set)

#####################################

# can we replace pca with oja's rule ?

w = np.absolute(np.random.normal(0.1, 0.05, size=(layer1, layer2)))

NUM_ITRS = 1
NUM_TRAIN_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 10000

for itr in range(NUM_ITRS):
  print itr
  for i in range(NUM_TRAIN_EXAMPLES):
    x = training_set[i]
    y = np.dot(x, w)
    wy = np.dot(w, y)
    d = x - wy

    w = np.clip(w + 1e-9 * d.reshape(layer1, 1) * y.reshape(1, layer2), 1e-9, 1.0)
    
    # if (i % 100 == 0):
    col_norm = np.average(w, axis = 0)
    col_norm = 0.1 / col_norm
    for j in range(layer2):
      w[:, j] *= col_norm[j]
  
print np.std(w), np.average(w), np.max(w), np.min(w)

#####################################

train_spks = np.zeros(shape=(NUM_TRAIN_EXAMPLES, layer2))
max_rates = np.zeros(layer2)
assignments = np.zeros(layer2)
for i in range(NUM_TRAIN_EXAMPLES):
  x = training_set[i]
  y = np.dot(x, w)
  train_spks[i] = y / np.average(y) # normalize it to how much the whole image spiked.
  for j in range(layer2):
    if max_rates[j] < y[j]:
      max_rates[j] = y[j]
      assignments[j] = training_labels[i]
  
#####################################
  
test_spks = np.zeros(shape=(NUM_TEST_EXAMPLES, layer2))
correct = 0
for i in range(NUM_TEST_EXAMPLES):
  x = testing_set[i]
  y = np.dot(x, w)
  test_spks[i] = y
  
  max_spikes = 0
  predict = 0
  for j in range(10):
    mask = assignments == j
    num_assignments = np.count_nonzero(mask)
    spikes = y / np.average(y) * mask
    
    if np.sum(spikes):
      spikes = np.sum(spikes) / num_assignments
    else:
      spikes = 0
      
    if max_spikes < spikes:
      max_spikes = spikes
      predict = j
      
  correct += predict == training_labels[i]

print (1.0 * correct) / NUM_TEST_EXAMPLES 

#####################################

logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_spks, training_labels)
score = logisticRegr.score(test_spks, testing_labels)
print(score)







