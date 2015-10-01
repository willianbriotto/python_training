import matplotlib.pyplot as plt
import numpy as np

# Complex sinewave using DFT

N = 500
k = 3
n = np.arange(-N/2, N/2)
s = np.exp(1j * 2 * np.pi * k * n / N)

# np.imag = imaginate part
plt.plot(n, np.real(s)) # real part
plt.axis([-N/2, N/2-1, -1, 1])
plt.xlabel('n')
plt.ylabel('amplitude')

plt.show()

