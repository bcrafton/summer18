
import numpy as np

XeAe = np.load('XeAe.npy')
np.savetxt("XeAe.csv", XeAe, delimiter=" ")

AeAi = np.load('AeAi.npy')
np.savetxt("AeAi.csv", AeAi, delimiter=" ")

AiAe = np.load('AiAe.npy')
np.savetxt("AiAe.csv", AiAe, delimiter=" ")

