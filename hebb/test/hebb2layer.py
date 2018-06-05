
import numpy as np
import math
import numpy as np
import cPickle as pickle
import gzip
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=1)
parser.add_argument('--examples', type=int, default=10000)
parser.add_argument('--hi', type=float, default=10.0)
parser.add_argument('--lo', type=float, default=0.1)
parser.add_argument('--lr', type=float, default=0.0001)
args = parser.parse_args()

layer1 = 28*28
layer2 = 40*40
layer3 = 20*20

print ("---------")
print (args.iters)
print (args.examples)
print (args.hi)
print (args.lo)
print (args.lr)

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

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

#####################################
load_data()
  
w1 = np.absolute(np.random.normal(1.0, 0.25, size=(layer1, layer2)))
w2 = np.absolute(np.random.normal(1.0, 0.25, size=(layer2, layer3)))
#####################################
prev1 = np.copy(w1)
prev2 = np.copy(w2)

for itr in range(args.iters):
  for i in range(args.examples):
    x = np.array(training_set[i]).reshape(layer1, 1)
    x = x / np.average(x)

    xw1 = np.dot(np.transpose(x), w1)
    xw1 = xw1 / np.max(xw1)
    xw1 = np.power(xw1, 3)
    
    sig = sigmoid(xw1)
    e = sig - 0.5 * xw1
    w1 += args.lr * np.dot(x, e)
    w1 = np.clip(w1, args.lo, args.hi)
    
    col_norm = np.average(w1, axis = 0)
    col_norm = 1.0 / col_norm
    for j in range(layer2):
      w1[:, j] *= col_norm[j]
    
    xw2 = np.dot(xw1, w2)
    xw2 = xw2 / np.max(xw2)
    xw2 = np.power(xw2, 3)
    
    sig = sigmoid(xw2)
    e = sig - 0.5 * xw2
    w2 += args.lr * np.dot(np.transpose(xw1), e)
    w2 = np.clip(w2, args.lo, args.hi)

    col_norm = np.average(w2, axis = 0)
    col_norm = 1.0 / col_norm
    for j in range(layer3):
      w2[:, j] *= col_norm[j]
      
    if (i % 999 == 0):
        # print (np.sum(np.absolute(w1 - prev1)), np.sum(w1), np.sum(np.absolute(w1 - prev1)) / np.sum(w1))
        prev1 = np.copy(w1)
        # print (np.sum(np.absolute(w2 - prev2)), np.sum(w2), np.sum(np.absolute(w2 - prev2)) / np.sum(w2))
        prev2 = np.copy(w2)

#####################################
max_rates1 = np.zeros(layer2)
assignments1 = np.zeros(layer2)

max_rates2 = np.zeros(layer3)
assignments2 = np.zeros(layer3)

for i in range(args.examples):
  x = np.array(training_set[i]).reshape(layer1, 1)
  x = x / np.average(x)

  xw1 = np.dot(np.transpose(x), w1)
  xw1 = xw1 / np.max(xw1)
  xw1 = np.power(xw1, 3)
  xw1 = xw1.flatten()

  for j in range(layer2):
    if max_rates1[j] < xw1[j]:
      max_rates1[j] = xw1[j]
      assignments1[j] = training_labels[i]

  xw2 = np.dot(xw1, w2)
  xw2 = xw2 / np.max(xw2)
  xw2 = np.power(xw2, 3)
  xw2 = xw2.flatten()
  
  for j in range(layer3):
    if max_rates2[j] < xw2[j]:
      max_rates2[j] = xw2[j]
      assignments2[j] = training_labels[i]

# print (assignments1)
# print (assignments2)
#####################################
correct = 0

for i in range(args.examples):
  x = np.array(training_set[i]).reshape(layer1, 1)
  x = x / np.average(x)

  xw1 = np.dot(np.transpose(x), w1)
  xw1 = xw1 / np.max(xw1)
  xw1 = np.power(xw1, 3)
  xw1 = xw1.flatten()
  
  xw2 = np.dot(xw1, w2)
  xw2 = xw2 / np.max(xw2)
  xw2 = np.power(xw2, 3)
  xw2 = xw2.flatten()

  max_spikes = 0
  for j in range(10):
    mask = assignments2 == j
    num_assignments = np.count_nonzero(mask)
    spikes = xw2 * mask
    
    if np.sum(spikes):
      spikes = np.sum(spikes) / num_assignments
    else:
      spikes = 0
      
    if max_spikes < spikes:
      max_spikes = spikes
      predict = j
  
  # print (predict, training_labels[i])
  correct += predict == training_labels[i]
  
print (np.std(w1))
print (np.std(w2))
print (1.0 * correct) / args.examples
#####################################

