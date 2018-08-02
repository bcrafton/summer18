
import numpy as np

w150 = np.load('./150k/XeAe_trained.npy')
w50 = np.load('./50k/XeAe_trained.npy')
w10 = np.load('./10k/XeAe_trained.npy')
w1 = np.load('./1k/XeAe_trained.npy')

w20 = np.load('./trained/XeAe_trained_20000.npy')
w15 = np.load('./trained/XeAe_trained_15000.npy')
w5 = np.load('./trained/XeAe_trained_5000.npy')
w3 = np.load('./trained/XeAe_trained_3000.npy')

print np.sum(np.absolute(w150 - w150))
print np.sum(np.absolute(w50 - w150))
print np.sum(np.absolute(w20 - w150))
print np.sum(np.absolute(w15 - w150))
print np.sum(np.absolute(w10 - w150))
print np.sum(np.absolute(w5 - w150))
print np.sum(np.absolute(w3 - w150))
print np.sum(np.absolute(w1 - w150))

