
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    def initialize(self, input_size: tuple, num_classes: int):
        pass

    def forward(self, X: np.ndarray) -> np.ndarray:
        pass

    def dfa(self, E: np.ndarray) -> tuple:
        pass

    def back_prob(self, E: np.ndarray) -> tuple:
        pass

