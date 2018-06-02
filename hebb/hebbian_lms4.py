
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

'''
for i in range(10000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.average(x)
  xw = x * w
  for j in range(28*28):
    sort = np.sort(xw[j, :], axis=0)
    argsort = np.argsort(xw[j, :], axis=0)
    if (np.sum(sort) > 200):
      sum = np.cumsum(sort)
      
      spike = sum <= 200
      no_spike = sum > 200
      # print(np.count_nonzero(spike))
      
      gradient = (spike * sort) * 0.01
      w[j, argsort] = np.clip(w[j, argsort] + gradient, 0.05, 5)
  
  if (i % 10 == 0):
    col_norm = np.average(w, axis = 0)
    col_norm = 0.5 / col_norm
    for j in range(num_output):
        w[:, j] *= col_norm[j]
'''

'''
for i in range(10000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.average(x)
  
  xw = np.dot(np.transpose(x), w)
  # print(np.max(xw), np.min(xw), np.std(xw))
  xw = xw / np.max(xw)
  # xw = np.exp(xw)
  xw = np.power(xw, 12)
  # print(np.max(xw), np.min(xw), np.std(xw))
  w += 0.001 * np.dot(x, xw)
  # print (np.amax( np.dot(x, xw), axis=0 ))
  
  if (i % 10 == 0):
    col_norm = np.average(w, axis = 1)
    col_norm = 0.5 / col_norm
    for j in range(num_output):
        w[:, j] *= col_norm[j]
'''

for i in range(10000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.max(x)

  xw = np.dot(np.transpose(x), w)
  xw = xw / np.max(xw)
  xw = np.power(xw, 2)

  stdp = np.copy(w)
  stdp = stdp / np.max(stdp)
  stdp = np.power(w, 2)
  
  w += 0.05 * np.dot(x, xw) * stdp

  col_norm = np.average(w, axis = 0)
  col_norm = 0.5 / col_norm
  for j in range(num_output):
      w[:, j] *= col_norm[j]

#####################################
max_rates = np.zeros(num_output)
assignments = np.zeros(num_output)

for i in range(2000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.max(x)
  
  xw = np.dot(np.transpose(x), w)
  xw = np.array(xw).flatten()
  
  for j in range(num_output):
    if max_rates[j] < xw[j]:
      max_rates[j] = xw[j]
      assignments[j] = training_labels[i]
      
print (assignments)
#####################################
correct = 0
for i in range(2000):
  x = np.array(training_set[i]).reshape(28*28, 1)
  x = x / np.max(x)
  
  xw = np.dot(np.transpose(x), w)

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
  
  print (predict, training_labels[i])
  
  correct += predict == training_labels[i]
  
print (1.0 * correct) / 2000
#####################################
  
    
  
