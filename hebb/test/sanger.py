
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
parser.add_argument('--train', type=float, default=0)
args = parser.parse_args()

layer1 = 28*28
layer2 = 20*20

print ("---------")
print (args.iters)
print (args.examples)
print (args.hi)
print (args.lo)
print (args.lr)
print (args.train)

#####################################

def update_elig(elig, grad):
  elig = np.copy(elig)
  grad = np.copy(grad)
  
  if True:
      elig = elig * 0.9
      elig = elig + grad
  elif False:
      elig = np.power(elig, 1.25)
      elig = elig + grad
      elig = elig / np.max(elig)
  return elig

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f)
  f.close()

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(layer1)
  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(layer1)
    
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

#####################################

load_data()

#####################################

w = np.absolute(np.random.normal(1.0, 0.25, size=(layer1, layer2)))

#####################################

if args.train:
  prev = np.copy(w)
  for itr in range(args.iters):
    for i in range(args.examples):
      elig = np.zeros(shape=(784, 400))
      
      x = np.array(training_set[i]).reshape(1, layer1)
      x = x / np.average(x)

      y = np.dot(x, w)
      y = y / np.max(y)
      wy = np.dot(w, np.transpose(y))
      d = x - np.transpose(wy) / np.average(wy) * np.average(x)
      elig_grad = np.dot(np.transpose(y), d)
      elig = update_elig(elig, np.transpose(elig_grad))
      
      grad = args.lr * elig
      w = np.clip(w + grad, args.lo, args.hi)

      col_norm = np.average(w, axis = 0)
      col_norm = 1.0 / col_norm
      for j in range(layer2):
        w[:, j] *= col_norm[j]
        
      if (i % 999 == 0):
          # print (np.sum(np.absolute(w - prev)), np.sum(w), np.sum(np.absolute(w - prev)) / np.sum(w))
          prev = np.copy(w)

#####################################
max_rates = np.zeros(layer2)
assignments = np.zeros(layer2)

for i in range(args.examples):
  x = np.array(training_set[i]).reshape(1, layer1)
  x = x / np.average(x)

  y = np.dot(x, w)
  y = y / np.max(y)
  y = np.array(y).flatten()
  
  for j in range(layer2):
    if max_rates[j] < y[j]:
      max_rates[j] = y[j]
      assignments[j] = training_labels[i]
      
for i in range(10):
  count = np.count_nonzero((assignments == i))
  print str(i) + ": " + str(count)    
    
#####################################
correct = 0

for i in range(args.examples):
  x = np.array(training_set[i]).reshape(1, layer1)
  x = x / np.average(x)

  y = np.dot(x, w)
  y = y / np.max(y)

  max_spikes = 0
  for j in range(10):
    mask = assignments == j
    num_assignments = np.count_nonzero(mask)
    spikes = y * mask
    
    if np.sum(spikes):
      spikes = np.sum(spikes) / num_assignments
    else:
      spikes = 0
      
    if max_spikes < spikes:
      max_spikes = spikes
      predict = j
  
  correct += predict == training_labels[i]

print "std: " + str((np.std(w)))
print "accuracy: " + str((1.0 * correct) / args.examples)
#####################################

