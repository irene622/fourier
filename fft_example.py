import numpy as np
import matplotlib.pyplot as plt

signal = np.array([1,3,1,2])
timepoints = np.arange(4)
# (t,s) = (0,1), (1,3), (2,1), (3,2)

fourier_mat = np.array([[1,1,1,1], 
                        [1,1j,-1,-1j],
                        [1,-1,1,-1],
                        [1,-1j,-1,1j]])
# Discrete Fourier Transform
dfs = np.matmul(fourier_mat, signal)
print(dfs) # [7, 0+1j, -3, 0-1j]

fft = np.fft.fft(signal)
print(fft)
fft = np.fft.fftshift(fft)
print(fft)

