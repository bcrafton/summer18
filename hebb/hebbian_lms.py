
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

w = np.absolute(np.random.normal(0.5, 0.25, size=(200, 1)))

x = np.random.normal(0.5, 0.25, size=(200, 1))

for i in range(1000000):  
  w[100:] = w[100:] * -1
  xw = np.dot(np.transpose(x), w)
  if (xw < 0):
      xw = 0
  sig = sigmoid(xw)
  w[100:] = w[100:] * -1
  
  e = sig - 0.5 * xw
  w = np.clip(w + .01 * e * x, 0, 1) 
  
  print (e, np.std(w), np.average(w))
