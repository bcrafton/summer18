
import numpy as np
import matplotlib.pyplot as plt


bp_w2_angles = np.load("angles.npy")
bp_w2_acc = np.load("acc.npy")

rand_w2_angles = np.load("angles1.npy")
rand_w2_acc = np.load("acc1.npy")

plt.rcParams.update({'font.size': 18})

plt.subplot(211)
plt.plot(bp_w2_angles, '.', label="DFA w/ BP W2")
plt.plot(rand_w2_angles, '.', label="DFA w/ rand W2")
plt.legend(loc='upper right')
plt.ylabel("Angle")

plt.subplot(212)
plt.plot(bp_w2_acc, '.', label="DFA w/ BP W2")
plt.plot(rand_w2_acc, '.', label="DFA w/ rand W2")
plt.legend(loc='lower right')
plt.ylabel("Accuracy")

plt.show()  
