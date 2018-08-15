
import tensorflow as tf
import numpy as np
import math
# from tensorflow.python.ops import gen_nn_ops
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

    def forward(self, X: np.ndarray):
        return tf.nn.max_pool(value=X, ksize=self.size, strides=self.stride, padding="SAME")
    
    def backward(self, AIN : np.ndarray, AOUT : np.ndarray, DIN : np.ndarray):
        # think both of these would work
        
        # return tf.multiply(tf.cast(AOUT > 0.0, dtype=tf.float32), 1.0)
        
        pool = tf.nn.max_pool(value=AIN, ksize=self.size, strides=self.stride, padding="SAME")
        return tf.multiply(tf.cast(pool > 0.0, dtype=tf.float32), 1.0)
        
