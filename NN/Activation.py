
import numpy as np
import tensorflow as tf

class Activation(object):
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass
        
class Sigmoid(Activation):
    def __init__(self):
        pass

    def forward(self, x : np.ndarray):
        return tf.sigmoid(x)

    def sigmoid_gradient(self, x : np.ndarray):
        sig = tf.sigmoid(x)
        return tf.multiply(sig, tf.subtract(1.0, sig))
        
    def gradient(self, x : np.ndarray):
        return tf.multiply(x, tf.subtract(1.0, x))
        
class Relu(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return tf.nn.relu(x)

    def gradient(self, x):
        return tf.multiply(tf.cast(x > 0.0, dtype=tf.float32), 1.0)
