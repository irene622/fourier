import numpy as np
import matplotlib.pyplot as plt

begintime = -1
endtime = 20
timelength = endtime - begintime
samplingfreq = 1000
samplinginterval = timelength / samplingfreq
timepoints = np.arange(begintime, endtime, samplinginterval)
num_samples = len(timepoints)
print(num_samples)

alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x <= 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(x) for x in timepoints]

# Discrete Fourier transform with Fast Fourier Transform
fft = np.fft.fft(signal)
print(len(fft))
print(fft[0], fft[1], fft[2], fft[-1])
fft_magnitude = abs(fft)

N = samplingfreq 
T = 16 
delta_omega = 2*np.pi / T
P = int(N / T)
R = 2 # degree of phase function of H

# Draw the signal graph
plt.title("the amplitude")
plt.ylabel("amplitude")
freq = np.fft.fftfreq(num_samples)
print(freq[:4])
plt.plot(freq, fft_magnitude)
plt.grid()

plt.savefig("fft_magnitude.png")