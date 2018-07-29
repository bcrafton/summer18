
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

scaler = StandardScaler()
scaler.fit(training_set)

training_set = scaler.transform(training_set)
testing_set = scaler.transform(testing_set)

pca = PCA(.95)
pca.fit(training_set)

# can see values i think: 
# http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
# print pca.singular_values_

# comment this out bc turns it into 339 array.
# training_set = pca.transform(training_set)
# testing_set = pca.transform(testing_set)

#####################################

# can we replace pca with oja's rule ?

w = np.absolute(np.random.normal(0.1, 0.05, size=(layer1, layer2)))

NUM_ITRS = 3
NUM_TRAIN_EXAMPLES = 50000
NUM_TEST_EXAMPLES = 10000

for itr in range(NUM_ITRS):
  print itr
  for i in range(NUM_TRAIN_EXAMPLES):
    # x = np.array(training_set[i]).reshape(layer1)
    # x = x / np.average(x)
    x = training_set[i]
    y = np.dot(x, w)
    # y = y / np.max(y)
    wy = np.dot(w, y)
    d = x - wy
    # w = w + 1e-5 * d.reshape(784, 1) * y.reshape(1, 400)
    w = np.clip(w + 1e-5 * d.reshape(784, 1) * y.reshape(1, 400), 0, 1.0)
  
print np.std(w), np.average(w), np.max(w), np.min(w)
  
train_spks = np.zeros(shape=(NUM_TRAIN_EXAMPLES, layer2))
for i in range(NUM_TRAIN_EXAMPLES):
  # x = np.array(training_set[i]).reshape(layer1)
  # x = x / np.average(x)
  x = training_set[i]
  y = np.dot(x, w)
  # y = y / np.max(y)
  train_spks[i] = y
  
test_spks = np.zeros(shape=(NUM_TEST_EXAMPLES, layer2))
for i in range(NUM_TEST_EXAMPLES):
  # x = np.array(testing_set[i]).reshape(layer1)
  # x = x / np.average(x)
  x = testing_set[i]
  y = np.dot(x, w)
  # y = y / np.max(y)
  test_spks[i] = y

# how do we do pca transform on these things ? 
# i guess we just use spike counts ?
# that really is our transform ...
# yeah and just turn learning off for the testing set.

#####################################

# this performs 91.71 without pca transformation ... wud like to see the assignments performance with pca
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_spks, training_labels)

score = logisticRegr.score(test_spks, testing_labels)
print(score)



