
import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, layers : tuple):
        self.num_layers = len(layers)
        self.layers = layers
        
    def num_params(self):
        param_sum = 0
        for ii in range(self.num_layers):
            l = self.layers[ii]
            param_sum += l.num_params()
        return param_sum
        
    def train(self, X, Y):
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
    
    def dfa(self, X, Y):
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
                D[ii] = l.dfa(A[ii-1], A[ii], E, E)
                gvs = l.dfa_gv(A[ii-1], A[ii], E, E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.dfa(X, A[ii], E, D[ii+1])
                gvs = l.dfa_gv(X, A[ii], E, D[ii+1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.dfa(A[ii-1], A[ii], E, D[ii+1])
                gvs = l.dfa_gv(A[ii-1], A[ii], E, D[ii+1])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars
    
    def predict(self, X):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X, dropout=False)
            else:
                A[ii] = l.forward(A[ii-1], dropout=False)
                
        return A[self.num_layers-1]
        
        
        
        
        
        
        
        
