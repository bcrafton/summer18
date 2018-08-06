
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def sigmoid_gradient(x):
  gz = sigmoid(x)
  ret = gz * (1 - gz)
  return ret
  
def relu(x):
  ret = x * (x > 0.0)
  return ret
  
def relu_gradient(x):
  ret = 1.0 * (x > 0.0)
  return ret
  
def add_bias(x):
  return np.append(x, 1)
  
def minus_bias(x):
  cpy = np.copy(x)
  return cpy[:-1]
  
class nn_td:
    def __init__(self, size, weights, alpha, gamma, lmda, bias):
        # check to make sure we have the right number of layers and weights
        self.num_layers = len(size)
        self.num_weights = len(weights)
        assert(self.num_layers-1 == self.num_weights)
        
        # check to make sure that the sizes of the layers matches up
        for ii in range(1, self.num_layers):
            if bias:
                shape1 = (size[ii-1]+1, size[ii])
                shape2 = np.shape(weights[ii-1])
            else:
                shape1 = (size[ii-1], size[ii])
                shape2 = np.shape(weights[ii-1])
                
            assert(shape1 == shape2)
        
        # set size
        self.size = size
        
        # state variables
        self.weights = weights
        self.e = [None] * (self.num_layers-1)
        for ii in range(self.num_layers-1):
            if bias:
                self.e[ii] = np.zeros(shape=(size[ii]+1))
            else:
                self.e[ii] = np.zeros(shape=(size[ii]))

        # constants
        self.alpha = alpha
        self.gamma = gamma
        self.lmda = lmda
        self.bias = bias
        
    def predict(self, x):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                A[ii] = add_bias(x) if self.bias else x
                Z[ii] = None
            elif ii == self.num_layers-1:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = relu(Z[ii])
            else:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = add_bias(relu(Z[ii])) if self.bias else relu(Z[ii])
                                
        return A[self.num_layers-1]

    
    def train(self, state, action, reward, value, prev_value):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
        D = [None] * self.num_layers
        G = [None] * self.num_layers
    
        # feedforward
        for ii in range(self.num_layers):
            if ii == 0:
                A[ii] = add_bias(state) if self.bias else state
                Z[ii] = None
            elif ii == self.num_layers-1:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = relu(Z[ii])
            else:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = add_bias(relu(Z[ii])) if self.bias else relu(Z[ii])
                
        # update eligbility
        for ii in range(self.num_layers-1):
            self.e[ii] = self.gamma * self.lmda * self.e[ii] + (A[ii] / np.max(A[ii]) * (1 - self.lmda))
        
        # backprop
        for ii in range(self.num_layers-1, 0, -1):
            if ii == self.num_layers-1:
                D[ii] = np.zeros(2)
                D[ii][action] = reward + self.gamma * value - prev_value
            else:
                D[ii] = np.dot(D[ii+1], np.transpose(self.weights[ii]))
                if self.bias:
                    D[ii] = D[ii][:-1]
                    D[ii] = D[ii] * relu_gradient(Z[ii])
            
            if self.bias:
                self.weights[ii-1] += self.alpha * np.dot(self.e[ii-1].reshape(self.size[ii-1]+1, 1), D[ii].reshape(1, self.size[ii]))
            else:
                self.weights[ii-1] += self.alpha * np.dot(self.e[ii-1].reshape(self.size[ii-1], 1), D[ii].reshape(1, self.size[ii]))
            
    def clear(self):
        self.e = [None] * (self.num_layers-1)
        for ii in range(self.num_layers-1):
            if self.bias:
                self.e[ii] = np.zeros(shape=(self.size[ii]+1))
            else:
                self.e[ii] = np.zeros(shape=(self.size[ii]))
                
    def stds(self):
        _stds = np.zeros(shape=(self.num_layers-1))
        for ii in range(self.num_layers-1):
            _stds[ii] = np.std(self.weights[ii])
            
        return _stds

    def maxs(self):
        _maxs = np.zeros(shape=(self.num_layers-1))
        for ii in range(self.num_layers-1):
            _maxs[ii] = np.max(self.weights[ii])
            
        return _maxs

    def avgs(self):
        _avgs = np.zeros(shape=(self.num_layers-1))
        for ii in range(self.num_layers-1):
            _avgs[ii] = np.average(self.weights[ii])
            
        return _avgs
        
