import numpy as np
import matplotlib.pyplot as plt
import torch

def function(t) :
    return t + t**2

N = 64
timepoints = torch.arange(0, 2*np.pi, (2*np.pi)/64, dtype=float) 
signal = torch.tensor([function(k) for k in timepoints])

fft = torch.fft.fft(signal)
fft_magnitude = abs(fft)

plt.title("Fig 3.3 Plot of fast Fourier transform coefficients.")
plt.plot(timepoints[1:], fft_magnitude[1:], ".")
plt.grid()
plt.savefig("fft_math3.png")