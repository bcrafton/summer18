
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    def initialize(self, input_size: tuple, num_classes: int):
        pass

    def forward(self, batch_size : int, X: np.ndarray) -> np.ndarray:
        pass

    def backward(self, batch_size : int, E: np.ndarray) -> tuple:
        pass

    def dfa(self, batch_size : int, E: np.ndarray) -> tuple:
        pass


