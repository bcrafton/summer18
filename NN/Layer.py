
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    def get_weights(self):
        pass

    def forward(self, X, dropout=False):
        pass

    def backward(self, AI, AO, DO):
        pass
        
    def gv(self, AI, AO, DO):
        pass

    def dfa(self, AI, AO, E, DO):
        pass

    def dfa_gv(self, AI, AO, E, DO):
        pass


