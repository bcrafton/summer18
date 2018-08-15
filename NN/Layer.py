
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    def get_weights(self):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def backward(self, E: np.ndarray) -> tuple:
        pass

    def dfa(self, E: np.ndarray) -> tuple:
        pass


