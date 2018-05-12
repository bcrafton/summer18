
import numpy as np
import matplotlib.cm as cmap
import time
import os.path
import scipy 
import cPickle as pickle
import brian_no_units  #import it to deactivate unit checking --> This should NOT be done for testing/debugging 
import brian as b
from struct import unpack
from brian import *
import gzip

ending = ''
n_input = 784
n_e = 400
n_i = n_e 

def get_matrix_from_file(fileName):
    offset = len(ending) + 4
    if fileName[-4-offset] == 'X':
        n_src = n_input                
    else:
        if fileName[-3-offset]=='e':
            n_src = n_e
        else:
            n_src = n_i
    if fileName[-1-offset]=='e':
        n_tgt = n_e
    else:
        n_tgt = n_i
    readout = np.load(fileName)
    print readout.shape, fileName
    value_arr = np.zeros((n_src, n_tgt))
    if not readout.shape == (0,):
        value_arr[np.int32(readout[:,0]), np.int32(readout[:,1])] = readout[:,2]
    return value_arr


AeAi = get_matrix_from_file('AeAi.npy')
AiAe= get_matrix_from_file('AiAe.npy')
XeAe = get_matrix_from_file('XeAe.npy')

np.save('./load/AeAi', AeAi)
np.save('./load/AiAe', AiAe)
np.save('./load/XeAe', XeAe)
