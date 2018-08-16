
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Convolution(Layer):
    def __init__(self, input_sizes, filter_sizes, filters, stride, padding, alpha, activation: Activation=None, last_layer=False):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        
        self.filters = filters
        
        # TODO
        self.stride = stride
        self.stride = [1,1,1,1]
        
        # TODO
        self.padding = padding
        self.padding = "SAME"
        
        self.alpha = alpha
        
        self.activation = activation
        self.last_layer = last_layer
        
    def get_weights(self):
        return self.filters
        
    def forward(self, X : np.ndarray, dropout=False):
        Z = tf.nn.conv2d(X, self.filters, self.stride, self.padding)
        A = self.activation.forward(Z)
        return A
        
    def backward(self, AI: np.ndarray, AO: np.ndarray, DO: np.ndarray):
        # apply activation gradient
        DO = tf.multiply(DO, self.activation.gradient(AO))
        
        # send this back
        # DI = tf.nn.conv2d_backprop_input(input_sizes=[BATCH_SIZE, 28, 28, 1], filter=W, out_backprop=Y, strides=[1,1,1,1], padding="SAME")
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.stride, padding="SAME")
        
        # update with this
        # DF = tf.nn.conv2d_backprop_filter(input=X, filter_sizes=[5, 5, 1, 16], out_backprop=Y, strides=[1,1,1,1], padding="SAME")        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.stride, padding="SAME")
        
        # update filters
        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        
        # return error wtr to input 
        return DI

    def dfa(self, AIN: np.ndarray, AOUT: np.ndarray, DIN: np.ndarray):
        pass
