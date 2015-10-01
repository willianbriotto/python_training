import matplotlib.pyplot as plt
import numpy as np

A = .8
f0 = 1000 # Frequecy
phi = np.pi/2 # Phase
fs = 44100
t = np.arange(-0.002, .002, 1.0/fs)
x = A * np.cos(2 * np.pi * f0 * t + phi) # Time function

plt.plot(t, x)
plt.axis([-.002, .002, -.8, .8])
plt.xlabel('time')
plt.ylabel('amplitude')

plt.show()
