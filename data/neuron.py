import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

'''
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
'''

def NFET_IDS(i0, k, kn, vth, l, vds, vgs):

    ut = 0.026;

    sub_sat = (vgs < vth) and (vds >= 4*ut)
    sub_off = (vgs < vth) and (vds < 4*ut)
    sat = (vgs > vth) and (vds >= vgs - vth)

    if (sub_sat):
        ids = (i0 * np.exp(k * vgs / ut))
    elif(sub_off):
        ids = (i0 * np.exp(k * vgs / ut)) * (1 - np.exp(-vds / ut))
    elif (sat):
        ids = 0.5 * (kn * (vgs - vth) ** 2) * (1 + l * vds)
    else:
        ids = (kn * (vgs - vth) * vds - ((vds ** 2)/2)) * (1 + l * vds)

    return ids


### reset
rst_vmem = np.genfromtxt('rst_vmem.csv',delimiter=',')
rst_vo2 = np.genfromtxt('rst_vo2.csv',delimiter=',')
rst_m12 = np.genfromtxt('rst_m12.csv',delimiter=',')
rst_m12[np.where(rst_m12 < 0)] = 0 

# m12_func = interpolate.bisplrep(rst_vmem, rst_vo2, rst_m12, kx=5, ky=5)
# m12_func = interpolate.interp2d(rst_vmem, rst_vo2, rst_m12, kind='linear')
x = np.transpose([rst_vmem, rst_vo2])
y = rst_m12
m12_func = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)

### leak
leak_vmem = np.genfromtxt('leak_vmem.csv',delimiter=',')
leak_m20 = np.genfromtxt('leak_m20.csv',delimiter=',')
leak_m20[np.where(leak_m20 < 0)] = 0 

m20_func = np.polyfit(leak_vmem, leak_m20, deg=5)
# m20_func = interpolate.interp1d(leak_vmem, leak_m20, kind='cubic')
# x = leak_vmem
# y = leak_m20
# m20_func = interpolate.LinearNDInterpolator(leak_vmem, leak_m20)

### inv slew
slew_vo1 = np.genfromtxt('slew_vo1.csv',delimiter=',')
slew_vo2 = np.genfromtxt('slew_vo2.csv',delimiter=',')
slew_co2 = np.genfromtxt('slew_co2.csv',delimiter=',')
slew_co2[np.where(slew_co2 < 0)] = 0 

# co2_func = interpolate.bisplrep(slew_vo1, slew_vo2, slew_co2, kx=3, ky=3)
# co2_func = interpolate.interp2d(slew_vo1, slew_vo2, slew_co2, kind='cubic')
x = np.transpose([slew_vo1, slew_vo2])
y = slew_co2
co2_func = interpolate.LinearNDInterpolator(x, y, fill_value=0.0)
# co2_func = interpolate.NearestNDInterpolator(x, y)

# print(np.average(y), np.std(y))
# plt.plot(slew_vo1, slew_co2)

### fb
fb_vmem = np.genfromtxt('fb_vmem.csv',delimiter=',')
fb_m7 = np.genfromtxt('fb_m7.csv',delimiter=',')
fb_vo1 = np.genfromtxt('fb_vo1.csv',delimiter=',')

# m7_func = np.polyfit(fb_vmem, fb_m7, deg=5)
# vo1_func = np.polyfit(fb_vmem, fb_vo1, deg=5)
m7_func = interpolate.interp1d(fb_vmem, fb_m7, kind='cubic')
vo1_func = interpolate.interp1d(fb_vmem, fb_vo1, kind='cubic')

# x = fb_vmem
# y = fb_m7
# m7_func = interpolate.LinearNDInterpolator(fb_vmem, fb_m7)

# x = fb_vmem
# y = fb_vo1
# vo1_func = interpolate.LinearNDInterpolator(x, y)


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
ico1s = np.zeros(steps)
m7s = np.zeros(steps)
m12s = np.zeros(steps)
m20s = np.zeros(steps)

C1 = 500e-15
C2 = 100e-15


'''
vmem = np.linspace(0, 1, 1000)
m20 = np.zeros(1000)
for i in range(1000):
    m20[i] = np.polyval(m20_func, vmem[i])
plt.plot(vmem, m20)
plt.show()
'''

########################
'''
vmem = np.linspace(0, 1, 1000)
m7 = m7_func(vmem)

# m7 = np.zeros(1000)
#for i in range(1000):
#    m7[i] = np.polyval(m7_func, vmem[i])

plt.plot(vmem, m7)
plt.show()
'''
########################

'''
vmem = np.linspace(0, 1, 1000)
vo1 = vo1_func(vmem)
plt.plot(vmem, vo1)
plt.show()
'''

'''
vmem = np.linspace(1, 1, 1000)
vo2 = np.linspace(0, 1, 1000)
m12 = m12_func(vmem, vo2)
for i in range(1000):
    m12[i] = NFET_IDS(5e-12, 0.4, 2e-6, 0.325, 0.03, vmem[i], vo2[i])
plt.plot(vo2, m12)
plt.show()
'''

'''
vmem = np.linspace(1, 1, 1000)
vo2 = np.linspace(0, 1, 1000)
m12 = m12_func(vmem, vo2)
'''

'''
vo1 = np.linspace(1, 1, 1000)
vo2 = np.linspace(0, 1, 1000)
co2 = co2_func(vo1, vo2)
plt.plot(vo2, co2)
plt.show()
'''

# x1 = np.linspace(1, 1, 100)
# x2 = np.linspace(0, 1, 100)
n = len(slew_vo1)
y = np.zeros(n)
for i in range(n):
    y[i] =  co2_func(slew_vo1[i], slew_vo2[i])

# plt.plot(slew_vo1, slew_co2)
#plt.plot(slew_vo1, y)
# plt.plot(slew_vo1, slew_co2, slew_vo1, y)
#plt.show()

########################

for i in range(steps):
    
    t = Ts[i]
    '''
    if (i % 100 == 0):
        print (t)
    '''
    if (t > 1e-4):
        iin = 1e-9
    else:
        iin = 0
    
    # m7s[i] = np.polyval(m7_func, vmem)
    m7s[i] = m7_func(vmem)
    
    # m12s[i] = interpolate.bisplev(vmem, vo2, m12_func)
    # m12s[i] = NFET_IDS(5e-12, 0.4, 2e-6, 0.325, 0.03, vmem, vo2)
    m12s[i] = m12_func(vmem, vo2)
    
    m20s[i] = np.polyval(m20_func, vmem)
    #m20s[i] = m20_func(vmem)
    
    # vo1 = np.polyval(vo1_func, vmem)
    vo1 = vo1_func(vmem)
    
    # dvdt = (1 / C2) * interpolate.bisplev(vo1, vo2, co2_func)  
    ico1 = co2_func(vo1, vo2)
    dvdt = (1 / C2) * ico1
    vo2 = vo2 + dvdt * dt
    # vo2 = min(max(vo2, 0.0), 1.0)
        
    icmem = (iin - m20s[i] + m7s[i] - m12s[i])
    dvdt = (1 / C1) * icmem
    vmem = vmem + dvdt * dt
    # vmem = min(max(vmem, 0.0), 1.0)
    
    vmems[i] = vmem
    vo1s[i] = vo1
    vo2s[i] = vo2
    iins[i] = iin
    icmems[i] = icmem
    ico1s[i] = ico1

# plt.plot(Ts, vmems, Ts, vo1s)
plt.plot(Ts, icmems, Ts, ico1s)
plt.show()


