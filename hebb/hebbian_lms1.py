
import numpy as np
import math
import numpy as np
import cPickle as pickle
import gzip
import time

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)

  f.close()

load_data()

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
num_output = 20*20
w = np.absolute(np.random.normal(1.0, 0.25, size=(28*28, num_output)))
x = np.absolute(np.random.normal(1.0, 0.25, size=(28*28, 1)))
inh = 28*28/2

#####################################
for itr in range(1):
  for i in range(50000):

    x = np.array(training_set[i]).reshape(28*28, 1)
    x = x / np.max(x)

    w[:, inh:] = w[:, inh:] * -1
    xw = np.dot(np.transpose(x), w)
    nonzero = xw >= 0
    xw = xw * nonzero
    sig = sigmoid(xw)
    w[:, inh:] = w[:, inh:] * -1

    e = sig - 0.5 * xw
    # print (np.shape(e), np.shape(x), np.shape(x * e), np.shape(np.dot(x, e)))
    w = np.clip(w + 0.01 * np.dot(x, e), 0, 10)

    col_norm = np.average(w, axis = 0)
    col_norm = 1.0 / col_norm
    for j in range(num_output):
        w[:, j] *= col_norm[j]
#####################################
max_rates = np.zeros(num_output)
assignments = np.zeros(num_output)
for i in range(10000):

  x = np.array(training_set[i]).reshape(28*28, 1)
    
  w[:, inh:] = w[:, inh:] * -1
  xw = np.dot(np.transpose(x), w)
  nonzero = xw >= 0
  xw = xw * nonzero
  sig = sigmoid(xw)
  w[:, inh:] = w[:, inh:] * -1

  xw = np.array(xw).flatten()
  for j in range(num_output):
    if max_rates[j] < xw[j]:
      max_rates[j] = xw[j]
      assignments[j] = training_labels[i]
print (assignments)
#####################################
correct = 0
for i in range(10000):
  x = np.array(training_set[i]).reshape(28*28, 1)
    
  w[:, inh:] = w[:, inh:] * -1
  xw = np.dot(np.transpose(x), w)
  nonzero = xw >= 0
  xw = xw * nonzero
  sig = sigmoid(xw)
  w[:, inh:] = w[:, inh:] * -1

  max_spikes = 0
  for j in range(10):
    mask = assignments == j
    spikes = xw * mask
    spikes = np.sum(spikes)
    if max_spikes < spikes:
      max_spikes = spikes
      predict = j
      
  correct += predict == training_labels[i]
  
print (1.0 * correct) / 10000
#####################################
  
  
    
  
