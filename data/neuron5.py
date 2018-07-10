import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import time
import random
from scipy.integrate import solve_ivp

MAKE_INTERPOLATOR = False

try:
    import cPickle as pickle
except ImportError:
    import pickle

print "loading data"

####################################

if MAKE_INTERPOLATOR:
    # independent
    indiveri_vmem = np.genfromtxt('indiveri_vmem.csv',delimiter=',')
    indiveri_vo2 = np.genfromtxt('indiveri_vo2.csv',delimiter=',')
    indiveri_is1 = np.genfromtxt('indiveri_is1.csv',delimiter=',')

    # dependent
    indiveri_vspk = np.genfromtxt('indiveri_vspk.csv',delimiter=',')
    indiveri_io2 = np.genfromtxt('indiveri_io2.csv',delimiter=',')
    indiveri_imem = np.genfromtxt('indiveri_imem.csv',delimiter=',')

    print "building io2 interpolator"

    x = np.transpose([indiveri_vmem, indiveri_vo2, indiveri_is1])
    y = indiveri_io2
    io2_func = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
    # io2_func = interpolate.NearestNDInterpolator(x, y)

    print "building imem interpolator"

    x = np.transpose([indiveri_vmem, indiveri_vo2, indiveri_is1])
    y = indiveri_imem
    imem_func = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
    # imem_func = interpolate.NearestNDInterpolator(x, y)
else:
    with open('io2_func.pkl', 'rb') as f:
        io2_func = pickle.load(f)
    with open('imem_func.pkl', 'rb') as f:
        imem_func = pickle.load(f)

####################################

dt = 1e-6
T = 1e-3
steps = int(T / dt);
Ts = np.linspace(0, T, steps)

vmem = 0
vo2 = 0

vmems = np.zeros(steps)
vo2s = np.zeros(steps)
iins = np.zeros(steps)

ico2s = np.zeros(steps)
icmems = np.zeros(steps)

C1 = 500e-15
C2 = 100e-15

########################
'''
t1 = time.time()
for i in range(steps):
    icmem = imem_func(random.uniform(0, 1), random.uniform(0, 1), 1e-9)
t2 = time.time()
print ("total time taken:" + str(t2 - t1))
'''
########################

print "starting sim"
start = time.time()

y0 = [0, 0, 0]

def deriv(t, y):
    vmem = y[0]
    vo2 = y[1]
    iin = y[2]
    
    # print (t, vmem, vo2)
    
    dvmem_dt = (1 / C1) * imem_func(vmem, vo2, iin)
    dvo2_dt = (1 / C2) * io2_func(vmem, vo2, iin)
    
    return [dvmem_dt, dvo2_dt, 0.0]

# can experiment with rtol, atol
# rtol=1e-3
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp

# there is also min_step
sol = solve_ivp(deriv, (0, 1e-4), y0, method='RK45')
vmem = sol.y[0, -1]
vo2 = sol.y[1, -1]
y0 = [vmem, vo2, 1e-10]
print y0
sol = solve_ivp(deriv, (1e-4, 1), y0, method='RK45')

Ts = sol.t
vmems = sol.y[0, :]
vo2s = sol.y[1, :]

print (np.shape(Ts))
print (np.shape(vmems))
print (np.shape(vo2s))

'''
for i in range(steps):
    
    t = Ts[i]
    
    if (t > 7e-4):
        iin = 1e-9
    elif (t > 4e-4):
        iin = 1e-9
    elif (t > 1e-4):
        iin = 1e-9
    else:
        iin = 0
    
    icmem = imem_func(vmem, vo2, iin)
    dvdt = (1 / C1) * icmem
    vmem = vmem + dvdt * dt
    # vmem = min(max(vmem, 0.0), 1.0)
    
    ico2 = io2_func(vmem, vo2, iin)
    dvdt = (1 / C2) * ico2
    vo2 = vo2 + dvdt * dt
    # vo2 = min(max(vo2, 0.0), 1.0)
    
    vmems[i] = vmem
    vo2s[i] = vo2
    iins[i] = iin
    ico2s[i] = ico2
    icmems[i] = icmem
'''

end = time.time()
print ("total time taken: " + str(end - start))

#######################

if MAKE_INTERPOLATOR:
    with open('io2_func.pkl', 'wb') as f:
        pickle.dump(io2_func, f)
    with open('imem_func.pkl', 'wb') as f:
        pickle.dump(imem_func, f)
    
#######################

# plt.subplot(2,2,1)
# plt.plot(Ts, ico2s, Ts, icmems)

plt.subplot(2,2,2)
plt.plot(Ts, vmems, Ts, vo2s)

# plt.subplot(2,2,3)

# plt.subplot(2,2,4)

plt.show()


#######################




