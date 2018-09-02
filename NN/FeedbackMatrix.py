
import tensorflow as tf
import numpy as np
import math

def make_feedback_matrix(input_size : int, output_size : int, sparse : int, rank : int):
    mat = np.zeros(shape=(input_size, output_size))
    
    sqrt_fan_in = np.sqrt(input_size)
    lo = -sqrt_fan_in
    hi = sqrt_fan_in
    
    for ii in range(rank):
        tmp1 = np.random.uniform(lo, hi, size=(self.input_size, 1))
        tmp2 = np.random.uniform(lo, hi, size=(1, self.output_size))
        mat += 
