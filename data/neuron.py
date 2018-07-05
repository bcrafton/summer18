import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

def NFET_IDS(I0, K, KN, VTH, L, VDS, VGS):

    UT = 0.026

    if (VGS <= VTH):
        if (VDS > 4 * UT):
            IDS = I0 * np.exp((K * VGS) / UT)
        else:
            IDS = I0 * np.exp((K * VGS) / UT) * (1 - np.exp(-VDS / UT))
            
    elif (VGS > VTH) and (VDS <= VGS - VTH):
        IDS = (KN * (VGS - VTH) * VDS - (VDS ** 2 / 2)) * (1 + L * VDS)
        
    elif (VGS > VTH) and (VDS > VGS - VTH):
        IDS = 0.5 * (KN * (VGS - VTH) ** 2) * (1 + L * VDS)
        
    else:
        print (VDS, VGS)
        assert(False)
        
    return IDS


### reset
rst_vmem = np.genfromtxt('rst_vmem.csv',delimiter=',')
rst_vo2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
rst_m12 = np.genfromtxt('rst_m12.csv',delimiter=',')
rst_m12[np.where(rst_m12 < 0)] = 0 
m12_func = interpolate.bisplrep(rst_vmem, rst_vo2, rst_m12, kx=5, ky=5)

'''
m12 = interpolate.bisplev(vmem, vo2, m12_func)
m12[np.where(m12 < 0)] = 0 
'''

### leak
leak_vmem = np.genfromtxt('leak_vmem.csv',delimiter=',')
leak_m20 = np.genfromtxt('leak_m20.csv',delimiter=',')
leak_m20[np.where(leak_m20 < 0)] = 0 
m20_func = np.polyfit(leak_vmem, leak_m20, deg=5)

'''
vmem = np.linspace(0, 1, 100)
m20 = np.polyval(m20_func, vmem)
plt.plot(vmem, m20)
plt.show()
'''

### inv slew

slew_vmem = np.genfromtxt('slew_vmem.csv',delimiter=',')
slew_vo2 = np.genfromtxt('slew_vo2.csv',delimiter=',')
slew_co2 = np.genfromtxt('slew_co2.csv',delimiter=',')
slew_co2[np.where(slew_co2 < 0)] = 0 
co2_func = interpolate.bisplrep(slew_vmem, slew_vo2, slew_co2, kx=5, ky=5)

### fb

fb_vmem = np.genfromtxt('fb_vmem.csv',delimiter=',')
fb_m7 = np.genfromtxt('fb_m7.csv',delimiter=',')
fb_vo1 = np.genfromtxt('fb_vo1.csv',delimiter=',')

m7_func = np.polyfit(fb_vmem, fb_m7, deg=5)
vo1_func = np.polyfit(fb_vmem, fb_vo1, deg=5)

####################################

dt = 1e-7
T = 1e-3
steps = int(T / dt);

vmem = 0
vo1 = 0
vo2 = 0

Ts = np.linspace(0, T, steps)
vmems = np.zeros(steps)
vo1s = np.zeros(steps)
vo2s = np.zeros(steps)
iins = np.zeros(steps)
icmems = np.zeros(steps)
m7s = np.zeros(steps)
m12s = np.zeros(steps)
m20s = np.zeros(steps)

C1 = 500e-15
C2 = 100e-15

for i in range(steps):
    
    t = Ts[i]
    # print (t)
    
    if (t > 1e-4):
        iin = 1e-9
    else:
        iin = 0
    
    m7s[i] = np.polyval(m7_func, vmem)
    # m12s[i] = interpolate.bisplev(vmem, vo2, m12_func)
    m12s[i] = NFET_IDS(5e-12, 0.4, 2e-6, 0.325, 0.03, vmem, vo2)
    m20s[i] = np.polyval(m20_func, vmem)
    
    vo1 = np.polyval(vo1_func, vmem)
    
    dvdt = (1 / C2) * interpolate.bisplev(vo1, vo2, co2_func)
    vo2 = vo2 + dvdt * dt
    vo2 = min(max(vo2, 0.0), 1.0)
        
    icmem = (iin - m20s[i] + m7s[i] - m12s[i])
    dvdt = (1 / C1) * icmem
    vmem = vmem + dvdt * dt
    vmem = min(max(vmem, 0.0), 1.0)
    
    vmems[i] = vmem
    vo1s[i] = vo1
    vo2s[i] = vo2
    iins[i] = iin
    icmems[i] = icmem

plt.plot(Ts, vo1s)
plt.show()


