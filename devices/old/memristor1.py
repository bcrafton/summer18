import numpy as np 
from scipy.signal.waveforms import square 
import pylab as plt

################################################################################
# memristor 
################################################################################
class memristor:
  def __init__(self, sim_t, dt=0.25):
    self.ts = t = np.arange(0, sim_t+dt, dt)
    self.n = len(t)
    self.r = np.zeros(len(t))
    self.v = np.zeros(len(t))
    self.i = np.zeros(len(t))
    self.phi = np.zeros(len(t)) 
    self.q = np.zeros(len(t))
    self.dt = dt
	
  def q_t(self, t):
    """ cal q(t) from i """
    if t > 0:
      self.q[t] = self.q[t-1] + self.dt*self.i[t]

  def phi_t_cubic_poly(self, t):   
    """ cal phi(t) from q """
    q = self.q[t]
    self.phi[t] = q + (q**3)/3.0

  def phi_t_pw(self, t):   
    """ cal phi(t) from q """
    B = 1
    r0 = 10
    r1 = 100

    q = self.q[t]
    if (q < -B):
      self.phi[t] = r1*q + (r1-r0)*B
    elif (q > B):
      self.phi[t] = r1*q + (r0-r1)*B
    else:
      self.phi[t] = r0*q

  def v_t(self, t):
    """ cal v(t) = dphi/dt """
    if t == 0:
      self.v[t] = (self.phi[t] - 0.0)/self.dt
    else:
      self.v[t] = (self.phi[t] - self.phi[t-1])/self.dt
 
  def r_t(self, t):
    """ cal v(t) = dphi/dt """
    if t == 0:
      self.r[t] = 1.0
    else:
      if self.q[t] == self.q[t-1]:
        self.r[t] = self.r[t-1]
      else:
        self.r[t] = (self.phi[t] - self.phi[t-1]) / (self.q[t] - self.q[t-1])
 

################################################################################
# input (charge controled memristor)
################################################################################

m = memristor(sim_t=4.2*np.pi, dt=0.025)
for i, t in enumerate(m.ts):
  A = 0.8
  m.i[i] = A*np.sin(t)

################################################################################
# Simulate
################################################################################
for i,t in enumerate(m.ts):
  m.q_t(i)
  m.phi_t_pw(i)
  m.v_t(i)
  m.r_t(i)

plt.subplot(2,2,1)
plt.plot(m.ts, m.r * m.i)

plt.subplot(2,2,2)
plt.plot(m.r * m.i, m.i)

plt.subplot(2,2,3)
plt.plot(m.ts, m.i)

plt.subplot(2,2,4)
plt.plot(m.ts, m.r)

plt.show()











