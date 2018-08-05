
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
  
class NNDFA:
    def __init__(self, size, weights, fb_weights, alpha, bias):
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
            shape2 = np.shape(fb_weights[ii-1])
            assert(shape1 == shape2)
        
        self.size = size
        self.weights = weights
        self.fb_weights = fb_weights
        self.alpha = alpha
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

    
    def train(self, x, y):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
        D = [None] * self.num_layers
        G = [None] * self.num_layers
    
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

        
        for ii in range(self.num_layers-1, 0, -1):
            if ii == self.num_layers-1:
                E = A[ii] - y
                D[ii] = E
            else:
                D[ii] = E * np.transpose(self.fb_weights[ii]) * relu_gradient(Z[ii])

            if self.bias:
                G[ii-1] = np.dot(A[ii-1].reshape(self.size[ii-1]+1, 1), D[ii].reshape(1, self.size[ii]))
            else:
                G[ii-1] = np.dot(A[ii-1].reshape(self.size[ii-1], 1), D[ii].reshape(1, self.size[ii]))
                
            self.weights[ii-1] -= self.alpha * G[ii-1]
                
                
                
                
                

        
