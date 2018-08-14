
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, layers : tuple):
        self.num_layers = len(layers)
        self.layers = layers
        
    def train(self, batch_size : int, X : np.ndarray, Y : np.ndarray):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(batch_size, X)
            else:
                A[ii] = l.forward(batch_size, A[ii-1])
            
        E = A[self.num_layers-1] - Y
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii] = l.backward(batch_size, A[ii-1], A[ii], E)
            elif (ii == 0):
                D[ii] = l.backward(batch_size, X, A[ii], D[ii+1])
            else:
                D[ii] = l.backward(batch_size, A[ii-1], A[ii], D[ii+1])
                
        ret = []
        return ret
    
    def predict(self, batch_size : int, X : np.ndarray):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(batch_size, X)
            else:
                A[ii] = l.forward(batch_size, A[ii-1])
                
        return A[self.num_layers-1]
