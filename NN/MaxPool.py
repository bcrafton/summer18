
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops
# return gen_nn_ops.max_pool_v2(value=X, ksize=self.size, strides=self.strides, padding="SAME")

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class MaxPool(Layer):
    def __init__(self, size, ksize, strides, padding):
        self.size = size
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def num_params(self):
        return 0

    def forward(self, X, dropout=False):
        return tf.nn.max_pool(X, ksize=self.ksize, strides=self.strides, padding=self.padding)
    
    def backward(self, AI, AO, DO):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad
        
    def gv(self, AIN, AOUT, DO):
        return []
        
    def dfa(self, AI, AO, E, DO):
        grad = gen_nn_ops.max_pool_grad(grad=AO, orig_input=AI, orig_output=AO, ksize=self.ksize, strides=self.strides, padding=self.padding)
        return grad
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
