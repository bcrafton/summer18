
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Convolution(Layer):
    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, stride, padding, alpha, activation: Activation, last_layer, sparse):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        
        if init_filters == "zero":
            self.filters = tf.Variable(tf.zeros(shape=self.filter_sizes))
        elif init_filters == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.filters = tf.Variable(tf.random_uniform(shape=self.filter_sizes, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
        else:
            assert(False)

        self.bias = tf.Variable(tf.zeros(shape=self.fout))
        
        sqrt_fan_out = math.sqrt(self.fout * self.h * self.w)
        if sparse:
            b = np.zeros(shape=(self.fout * self.h * self.w, self.num_classes))
            for ii in range(self.fout * self.h * self.w):
                idx = int(np.random.randint(0, self.num_classes))
                b[ii][idx] = np.random.uniform(-1.0/sqrt_fan_out, 1.0/sqrt_fan_out)
            b = np.transpose(b)
            self.B = tf.cast(tf.Variable(b), tf.float32)
        else:
            self.B = tf.Variable(tf.random_uniform(shape=[self.num_classes, self.fout * self.h * self.w], minval=-1.0/sqrt_fan_out, maxval=1.0/sqrt_fan_out))

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
        
    def forward(self, X, dropout=False):
        Z = tf.add(tf.nn.conv2d(X, self.filters, self.stride, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        A = self.activation.forward(Z)
        return A
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.stride, padding="SAME")
        return DI

    def gv(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.stride, padding="SAME")
        return [(DF, self.filters)]

    def dfa(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        E = tf.matmul(E, self.B)
        E = tf.reshape(E, [self.batch_size, self.h, self.w, self.fout])
        E = tf.multiply(E, self.activation.gradient(AO))
        E = tf.multiply(E, DO)
        
        # dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(E)) > 0.5, tf.float32)
        # E = E * dropout_mask
        
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=E, strides=self.stride, padding="SAME")
        DB = tf.reduce_sum(E, axis=[0, 1, 2])
        
        return [(DF, self.filters), (DB, self.bias)]
        
        
        
        
        
        
        
