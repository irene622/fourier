import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

begintime = 0
endtime = 16
timelength = endtime - begintime
N = 100 # number of sample
sampling_freq = timelength / N
timepoints = np.arange(begintime, endtime, sampling_freq)
l = 4
T = 2 ** l # period
delta_omega = 2*np.pi / T
P = int(N / T) # number of period
R = 2 # degree of phase function of lanmda_T

alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x <= 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(x) for x in timepoints]

# make the group delay of signal
fft = np.fft.fft(signal)
fft = np.fft.fftshift(fft)
fft_magnitude = abs(fft)
# fft = fft / fft_magnitude
angle = np.angle(fft)
angle = angle[:int(N/2)-1]
fft_magnitude = fft_magnitude[:int(N/2)]

T = N / sampling_freq
k = np.arange(N)
freq = k / T
freq = freq[:int(N/2)]
print(freq)
f = interpolate.interp1d(freq, fft_magnitude, kind = 'quadratic')

# Draw the signal graph
plt.figure(figsize=(7,5))
plt.title("f")

plt.plot(freq, f(freq))
plt.grid()

plt.savefig("04group_delay_signal.png")