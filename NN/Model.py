
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, layers : tuple):
        self.num_layers = len(layers)
        self.layers = layers
        
    def train(self, X : np.ndarray, Y : np.ndarray):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X, dropout=True)
            else:
                A[ii] = l.forward(A[ii-1], dropout=True)
            
        E = A[self.num_layers-1] - Y
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii] = l.backward(A[ii-1], A[ii], E)
                gvs = l.gv(A[ii-1], A[ii], E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii], D[ii+1])
                gvs = l.gv(X, A[ii], D[ii+1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.backward(A[ii-1], A[ii], D[ii+1])
                gvs = l.gv(A[ii-1], A[ii], D[ii+1])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars
    
    def dfa(self, X : np.ndarray, Y : np.ndarray):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X, dropout=True)
            else:
                A[ii] = l.forward(A[ii-1], dropout=True)
            
        E = A[self.num_layers-1] - Y
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
                
            if (ii == self.num_layers-1):
                D[ii] = l.dfa(A[ii-1], A[ii], E, E)
            elif (ii == 0):
                D[ii] = l.dfa(X, A[ii], E, D[ii+1])
            else:
                D[ii] = l.dfa(A[ii-1], A[ii], E, D[ii+1])
                
        ret = []
        for ii in range(self.num_layers):
            l = self.layers[ii]
            ret.append(l.get_weights())
        return ret
    
    def predict(self, X : np.ndarray):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X, dropout=False)
            else:
                A[ii] = l.forward(A[ii-1], dropout=False)
                
        return A[self.num_layers-1]
        
        
        
        
        
        
        
        
