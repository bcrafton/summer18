
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

layer1 = 28*28
layer2 = 20*20

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

for ii in range(50000):
    perm = np.random.permutation(784)
    training_set[ii][:] = training_set[ii][perm]

pca = PCA(.95)
pca.fit(training_set)

# print np.shape(training_set)
print pca.n_components_

training_set = pca.transform(training_set)
testing_set = pca.transform(testing_set)

logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(training_set, training_labels)

score = logisticRegr.score(testing_set, testing_labels)
print(score)

mat = np.random.uniform(low=-1.0, high=1.0, size=(10000, 785*25 + 26*10))
pca.fit(mat)
print pca.n_components_








