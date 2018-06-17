
import numpy as np
import matplotlib.pyplot as plt

class capacitor:
  def __init__(self, C, T, dt):
    self.C = C
    self.dt = dt
    self.T = T
    self.t = 0
    self.timesteps = int(self.T / self.dt)
    
    self.q = np.zeros(self.timesteps)
    self.v = np.zeros(self.timesteps)
    self.i = np.zeros(self.timesteps)
    self.z = np.ones(self.timesteps) * 10.0 

  def step(self, vin):
    self.i[self.t] = np.clip(1.0 * vin / self.z[self.t], 1e-200, 10)
    self.q[self.t] = self.i[self.t] * self.dt
    self.v[self.t] = self.q[self.t] / self.C
    if self.t < self.timesteps-1:
      self.z[self.t+1] = 1 / self.i[self.t]
      self.t = self.t + 1
    else:
      print("sim finished")
    
cap = capacitor(1, 1, 0.001)
vin = np.linspace(0, 1, 1000)
for i in range(1000):
  cap.step(vin[i])

plt.plot(vin, cap.z)
plt.show()


