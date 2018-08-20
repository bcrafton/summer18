
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    def get_weights(self):
        pass

    def forward(self, X : np.ndarray, dropout=False):
        pass

    def backward(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
        pass
        
    def gv(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
        pass

    def dfa(self, AI: np.ndarray, AO: np.ndarray, E: np.ndarray, DO: np.ndarray):
        pass


