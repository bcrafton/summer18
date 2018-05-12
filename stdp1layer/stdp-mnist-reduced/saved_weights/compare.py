import numpy as np

ref = np.load('ref_XeAe.npy')
x = np.load('XeAe.npy')
res = ref - x

print np.average(ref), np.average(x)

ref = np.load('ref_theta_A.npy')
x = np.load('theta_A.npy')
res = ref - x

print np.average(ref), np.average(x)

# print np.all(res == 0)
# print np.count_nonzero(res)
