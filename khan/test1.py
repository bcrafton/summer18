
import numpy as np
import matplotlib.pyplot as plt

T = 0.35
dt = 1e-2
steps = int(T / dt)
Ts = np.linspace(0, T, steps)

pre = 1
pres = []

for i in range(steps):
    pres.append(pre)
    
    t = Ts[i]
    dpre = -pre / 20e-3 * dt
    pre += dpre
  
pres = np.array(pres) 

pre = 1
pres1 = pre * np.exp(-Ts / 20e-3)
  
plt.plot(Ts, pres1, 'o')
plt.show()

