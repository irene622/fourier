import numpy as np
import matplotlib.pyplot as plt
 
begintime = -np.pi
endtime = np.pi
samplingfreq = 50
samplinginterval = 1 / samplingfreq
timepoints = np.arange(begintime, endtime, samplinginterval)
num_samples = len(timepoints)
print(f"the number of samples: {num_samples}") # 샘플의 갯수

# create signal
signal = timepoints - timepoints + 1
 
# Discrete Fourier transform with Fast Fourier Transform
fft = np.fft.fft(signal) / len(timepoints) # normalize
print(f"fft: {fft[:10]}")
 
fft_magnitude = abs(fft)

# create subplot
figure, axis = plt.subplots(2, 1, figsize=(8, 8))
plt.subplots_adjust(hspace=1)

# Draw the signal graph
axis[0].set_title("the signal")
axis[0].set_xlabel("time")
axis[0].plot(timepoints, signal)
axis[0].grid()

# Draw the fft in real part
axis[1].set_title("fft magnitude")
axis[1].set_xlabel("frequency")
freq = np.fft.fftfreq(num_samples)
axis[1].plot(freq, fft_magnitude)
axis[1].grid()
 
plt.savefig("savefig_default2.png")
 
