import numpy as np
import matplotlib.pyplot as plt

begintime = 0
endtime = 10
timelength = endtime - begintime
samplingfreq = 512
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
fft = np.fft.fftshift(fft)
fft_magnitude = abs(fft)

# freq domain
k = np.arange(num_samples)
Fs = samplinginterval
T = num_samples / Fs
freq = k / T

# Draw the signal graph
plt.title("the amplitude")
plt.ylabel("amplitude")
plt.plot(freq, fft_magnitude)
plt.grid()

plt.savefig("02fft_magnitude.png")