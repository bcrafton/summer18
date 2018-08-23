
import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops import gen_nn_ops
# return gen_nn_ops.max_pool_v2(value=X, ksize=self.size, strides=self.stride, padding="SAME")

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class MaxPool(Layer):
    def __init__(self, size, stride):
        self.size = size
        self.stride = stride

    def get_weights(self):
        return tf.random_uniform(shape=(1, 1))

    def forward(self, X, dropout=False):
        return tf.nn.max_pool(X, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    
    def backward(self, AI, AO, DO):
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        return grad
        
    def gv(self, AIN, AOUT, DO):
        return []
        
    def dfa(self, AI, AO, E, DO):
        grad = gen_nn_ops.max_pool_grad(grad=AO, orig_input=AI, orig_output=AO, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
        return grad
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
