
import tensorflow as tf
import numpy as np

class Network:
    def __init__(self):
        self.model = None
        
    def num_params(self):
        return self.model.num_params()
        
    def train(self, X, Y):
        return self.model.train(X, Y)
    
    def dfa(self, X, Y):
        return self.model.dfa(X, Y)
    
    def predict(self, X):
        return self.model.predict(X)
        
        
        
        
        
        
        
        
