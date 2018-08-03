
import numpy as np
import math

def sigmoid(x):
  return 1 / (1 + np.exp(-x))
  
def sigmoid_gradient(x):
  gz = sigmoid(x)
  ret = gz * (1 - gz)
  return ret
    
class NN:
    def __init__(self, size, weights, alpha, bias):
        # check to make sure we have the right number of layers and weights
        self.num_layers = len(size)
        self.num_weights = len(weights)
        assert(self.num_layers-1 == self.num_weights)
        
        # check to make sure that the sizes of the layers matches up
        for ii in range(1, self.num_layers):
            if bias and ii < self.num_layers-1:
                shape1 = (size[ii-1]+1, size[ii])
                shape2 = np.shape(weights[ii-1])
            else:
                shape1 = (size[ii-1]+1, size[ii])
                shape2 = np.shape(weights[ii-1])
            assert(shape1 == shape2)
        
        self.size = size
        self.weights = weights
        self.alpha = alpha
        self.bias = bias
        
    def predict(self, x):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                A[ii] = np.append(x, 1)
                Z[ii] = None
            elif ii == self.num_layers-1:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = sigmoid(Z[ii])
            else:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = np.append(sigmoid(Z[ii]), 1)
                
        return A[self.num_layers-1]

    
    def train(self, x, y):
        A = [None] * self.num_layers
        Z = [None] * self.num_layers
        D = [None] * self.num_layers
        G = [None] * self.num_layers
    
        for ii in range(self.num_layers):
            if ii == 0:
                A[ii] = np.append(x, 1)
                Z[ii] = None
            elif ii == self.num_layers-1:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = sigmoid(Z[ii])
            else:
                Z[ii] = np.dot(A[ii-1], self.weights[ii-1])
                A[ii] = np.append(sigmoid(Z[ii]), 1)
                
        # if u write things out, then this becomes much easier.
        # D = [3,2,1]
        # G = [2,1,0]
        # A = [3,2,1,0]
        # Z = [3,2,1]
        # W = [2,1,0]
        # really shuda wrote this out, probably still a better way to code this.
        # gradient corresponds to weights, A, Z, D correspond to neurons
        # think it through urself u can also derive it.
        
        for ii in range(self.num_layers-1, 0, -1):

            if ii == self.num_layers-1:
                D[ii] = A[ii] - y
            else:
                D[ii] = np.dot(D[ii+1], np.transpose(self.weights[ii])) * np.append(sigmoid_gradient(Z[ii]), 1)
                if self.bias:
                    D[ii] = D[ii][:-1]

            G[ii-1] = np.dot(A[ii-1].reshape(self.size[ii-1]+1, 1), D[ii].reshape(1, self.size[ii]))
            self.weights[ii-1] -= self.alpha * G[ii-1]
                
                
                
                
                

        
