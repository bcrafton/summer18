import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

print "loading data"

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

####################################

dt = 1e-6
T = 1e-2
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

print "starting sim"

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

plt.subplot(2,2,1)
plt.plot(Ts, ico2s, Ts, icmems)

plt.subplot(2,2,2)
plt.plot(Ts, vmems, Ts, vo2s)

# plt.subplot(2,2,3)

# plt.subplot(2,2,4)

plt.show()


