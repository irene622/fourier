import numpy as np
import matplotlib.pyplot as plt

begintime = 0
endtime = 32
timelength = endtime - begintime
N = 512 # number of sample
sampling_freq = timelength / N # sampling points들의 간격.
timepoints = np.arange(begintime, endtime, sampling_freq)

alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x <= 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(1*x) for x in timepoints]

# Discrete Fourier transform with Fast Fourier Transform
fft = np.fft.fft(signal)
fft = np.fft.fftshift(fft)
fft_magnitude = abs(fft)

# freq domain
T = 16
k = np.arange(-(T/2), T/2, T/N)
freq = k / T
k = np.arange(0, 8*np.pi, 8*np.pi/N) 

# Draw the signal graph
plt.title("the amplitude")
plt.ylabel("amplitude")
plt.plot(freq, fft_magnitude)
# plt.plot([2*np.pi/3, 8*np.pi/3], [0,0], "^")
plt.grid()

plt.savefig("02fft_magnitude.png")