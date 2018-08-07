
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
  
class nn_sarsa_dfa:
    def __init__(self, size, weights, b, alpha, gamma, lmda, bias):
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
            
        # check to make sure that the sizes of the feed back layers matches up
        for ii in range(1, self.num_layers):
            shape1 = (size[ii-1], size[self.num_layers-1])
            shape2 = np.shape(b[ii-1])
            assert(shape1 == shape2)
                
        # state variables
        self.weights = weights
        self.b = b
        self.e = [None] * (self.num_layers-1)
        for ii in range(self.num_layers-1):
            self.e[ii] = np.zeros(shape=(size[ii]+1, size[ii+1]))
        
        # constants
        self.size = size        
        self.alpha = alpha
        self.gamma = gamma
        self.lmda = lmda
        self.bias = bias
        
    def predict(self, state):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                Z[ii] = None
                A[ii] = add_bias(state) if self.bias else state
                # A0 may be less than 0
                assert(np.all(A[ii] >= 0.0))
            elif ii == self.num_layers-1:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = relu(Z[ii])
                assert(np.all(A[ii] >= 0.0))
            else:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = add_bias(relu(Z[ii])) if self.bias else relu(Z[ii])
                assert(np.all(A[ii] >= 0.0))
                
        return A[self.num_layers-1]

    
    def train(self, state, action, target):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
        D = [None] * self.num_layers
        G = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                Z[ii] = None
                A[ii] = add_bias(state) if self.bias else state
                # A0 may be less than 0
                assert(np.all(A[ii] >= 0.0))
            elif ii == self.num_layers-1:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = relu(Z[ii])
                assert(np.all(A[ii] >= 0.0))
            else:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = add_bias(relu(Z[ii])) if self.bias else relu(Z[ii])
                assert(np.all(A[ii] >= 0.0))
                
        for ii in range(self.num_layers-1, 0, -1):
            if ii == self.num_layers-1:
                E = A[ii] - target
                if action:
                    D[ii] = A[ii]
                else:
                    D[ii] = np.zeros(shape=np.shape(A[ii]))
            else:
                D[ii] = np.dot(D[ii+1], np.transpose(self.b[ii])) * relu_gradient(Z[ii])

            G[ii-1] = np.dot(A[ii-1].reshape(self.size[ii-1]+1, 1), D[ii].reshape(1, self.size[ii]))
            self.e[ii-1] = self.gamma * self.lmda * self.e[ii-1] + G[ii-1]
            self.weights[ii-1] -= self.alpha * E * self.e[ii-1]
                
    def clear(self):
        self.e = [None] * (self.num_layers-1)
        for ii in range(self.num_layers-1):
            self.e[ii] = np.zeros(shape=(self.size[ii]+1, self.size[ii+1]))
                

        
