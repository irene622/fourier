### Example 1 : FFT in numpy
import numpy as np
import matplotlib.pyplot as plt

signal = np.array([1,3,1,2])
timepoints = np.arange(4)
# (t,s) = (0,1), (1,3), (2,1), (3,2)

fourier_mat = np.array([[1,1,1,1], 
                        [1,-1j,-1,1j],
                        [1,-1,1,-1],
                        [1,1j,-1,-1j]])
# Discrete Fourier Transform
dfs = np.matmul(fourier_mat, signal)
print(dfs) # [7, 0+1j, -3, 0-1j]

fft = np.fft.fft(signal)
print(fft) # [ 7.+0.j  0.-1.j -3.+0.j  0.+1.j]
fft = np.fft.fftshift(fft)
print(fft) # [-3.+0.j  0.+1.j  7.+0.j  0.-1.j]



### Example 2 : FFT in torch
import torch

signal = torch.tensor([1,3,1,2], dtype=torch.complex64)
timepoints = torch.arange(4)
# (t,s) = (0,1), (1,3), (2,1), (3,2)

fourier_mat = torch.tensor([[1,1,1,1], 
                            [1,-1j,-1,1j],
                            [1,-1,1,-1],
                            [1,1j,-1,-1j]], dtype=torch.complex64)

# Discrete Fourier Transform
dfs = torch.matmul(fourier_mat, signal)
print(dfs) # [7, 0+1j, -3, 0-1j]
fft = torch.fft.fft(signal)
print(fft) # tensor([ 7.+0.j,  0.-1.j, -3.+0.j,  0.+1.j])



### Example 3 : Experiment Example 3.5 in [A First Course in Wavelets with Fourier Analysis-Wiley], Albert Boggess, 2009
def function(t) :
    return t + t^2

N = 64
timepoints = torch.arange(N)
signal = torch.tensor([function(k) for k in timepoints])

fft = torch.fft.fft(signal) / 8
fft_magnitude = abs(fft)

plt.title("Fig 3.3 Plot of fast Fourier transform coefficients.")
plt.plot(timepoints, fft_magnitude, ".")
plt.grid()
plt.savefig("fft_math3.png")
