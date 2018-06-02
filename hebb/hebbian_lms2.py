
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
w = np.absolute(np.random.normal(0.5, 0.25, size=(28*28, num_output)))
x = np.absolute(np.random.normal(0.5, 0.25, size=(28*28, 1)))

#####################################
prev = np.copy(w)
for i in range(50000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.average(x)

  xw = np.dot(np.transpose(x), w)
  xw = xw / np.max(xw)
  xw = np.power(xw, 3)
  
  sig = sigmoid(xw)
  e = sig - 0.25 * xw
  w += 0.0001 * np.dot(x, e)

  col_norm = np.average(w, axis = 0)
  col_norm = 0.5 / col_norm
  for j in range(num_output):
    w[:, j] *= col_norm[j]
    
  if (i % 1000 == 0):
      print (np.sum(np.absolute(w - prev)), np.sum(w), np.sum(np.absolute(w - prev)) / np.sum(w))
      prev = np.copy(w)
#####################################
max_rates = np.zeros(num_output)
assignments = np.zeros(num_output)

for i in range(5000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.average(x)
  
  xw = np.dot(np.transpose(x), w)
  xw = xw / np.max(xw)
  xw = np.power(xw, 2)
  xw = np.array(xw).flatten()
  
  for j in range(num_output):
    if max_rates[j] < xw[j]:
      max_rates[j] = xw[j]
      assignments[j] = training_labels[i]
      
print (assignments)
#####################################
correct = 0

for i in range(5000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.average(x)
  
  xw = np.dot(np.transpose(x), w)
  xw = xw / np.max(xw)
  xw = np.power(xw, 2)

  max_spikes = 0
  for j in range(10):
    mask = assignments == j
    num_assignments = np.count_nonzero(mask)
    spikes = xw * mask
    
    if np.sum(spikes):
      spikes = np.sum(spikes) / num_assignments
    else:
      spikes = 0
      
    if max_spikes < spikes:
      max_spikes = spikes
      predict = j
  
  # print (predict, training_labels[i])
  correct += predict == training_labels[i]
  
print (1.0 * correct) / 5000
#####################################

