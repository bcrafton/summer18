
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

    def forward(self, X: np.ndarray):
        # return tf.nn.max_pool(value=X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        return tf.nn.max_pool(X, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME")
    
    def backward(self, AI : np.ndarray, AO : np.ndarray, DO : np.ndarray):
        # think both of these would work
        
        # return tf.multiply(tf.cast(AOUT > 0.0, dtype=tf.float32), 1.0)
        
        grad = gen_nn_ops.max_pool_grad(grad=DO, orig_input=AI, orig_output=AO, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
        # pool, argmax = tf.nn.max_pool_with_argmax(AIN, self.size, self.stride, padding="SAME")
        # return tf.multiply(tf.cast(pool > 0.0, dtype=tf.float32), 1.0)
        return grad
