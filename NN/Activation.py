
import numpy as np
import tensorflow as tf

class Activation(object):
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    def gradient(self, x: np.ndarray) -> np.ndarray:
        pass
        
class Sigmoid(Activation):
    def __init__(self) -> None:
        pass

    def forward(self, x : np.ndarray) -> np.ndarray:
        return tf.sigmoid(x)

    def sigmoid_gradient(self, x : np.ndarray) -> np.ndarray:
        sig = tf.sigmoid(x)
        return tf.multiply(sig, tf.subtract(1.0, sig))
        
    def gradient(self, x : np.ndarray) -> np.ndarray:
        return tf.multiply(x, tf.subtract(1.0, x))
        
class ReLU(Activation):

    def __init__(self) -> None:
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0, x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return x > 0
