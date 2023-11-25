import numpy as np
import matplotlib.pyplot as plt

begintime = -1
endtime = 10
timelength = endtime - begintime
samplingfreq = 512
samplinginterval = timelength / samplingfreq
timepoints = np.arange(begintime, endtime, samplinginterval)
num_samples = len(timepoints)

alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x <= 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(x) for x in timepoints]

fft = np.fft.fft(signal)
real_part = fft.real
freq = np.fft.fftfreq(num_samples)

# Draw the signal graph
plt.title("signal")
plt.xlabel("timepoints")
plt.plot(timepoints, signal)
plt.grid()

plt.savefig("draw_signal.png")