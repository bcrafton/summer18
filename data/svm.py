import scipy
from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

x1 = np.genfromtxt('rst_vmem.csv',delimiter=',')
x2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
# x = np.transpose([x1, x2])
x = np.transpose([x1, x2])

y = np.genfromtxt('rst_m12.csv',delimiter=',')

print (np.shape(x))
print (np.shape(y))

clf = SVC()
clf.fit(x, y) 

'''
X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
print (np.shape(X))
y = np.array([1, 1, 2, 2])
print (np.shape(y))
'''

# plt.plot(x1, y)
# plt.show()

'''
def NFET_IDS(X, VTH, KSUB, K, UT, I0):

    print np.size(X)
    IDSS = np.zeros(shape=np.shape(X))

    for ii in range(len(X)):
    
        VGS = X[ii, 0]
        VDS = X[ii, 1]
    
        if (VGS <= VTH):
            if (VDS > 4 * UT):
                IDS = I0 * np.exp((KSUB * VGS) / UT)
            else:
                IDS = I0 * np.exp((KSUB * VGS) / UT) * (1 - np.exp(-VDS / UT))
                
        elif (VGS > VTH) and (VDS <= VGS - VTH):
            IDS = (K * (VGS - VTH) * VDS - (VDS ** 2 / 2)) * (1 + 0.01 * VDS)
            
        elif (VGS > VTH) and (VDS > VGS - VTH):
            IDS = 0.5 * (K * (VGS - VTH) ** 2) * (1 + 0.01 * VDS)
            
        else:
            print "should not get here"
            assert(False)
            
        IDSS[ii] = IDS
        
    return IDSS
    
######################
######################
    
scipy.optimize.curve_fit(NFET_IDS, x, y)
'''












