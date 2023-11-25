import numpy as np
import matplotlib.pyplot as plt
 
begintime = 0
endtime = np.pi*3
timelength = endtime - begintime
samplingfreq = 32
samplinginterval = timelength / samplingfreq
timepoints = np.arange(begintime, endtime+samplinginterval, samplinginterval)
num_samples = len(timepoints)
print(timepoints)
print(f"the number of samples: {num_samples}") # 샘플의 갯수

# create signal
# signal = timepoints - timepoints + 1
signal = np.ones(num_samples)
first = int(num_samples/3)
signal[:first] = 0
signal[first*2:] = 0
 
# Discrete Fourier transform with Fast Fourier Transform
fft = np.fft.fft(signal) # 결과 체크를 하기. 작은 샘플량으로도 해보기.
print(f"fft: {fft}")
 
fft_magnitude = abs(fft)

# create subplot
figure, axis = plt.subplots(3, 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.5)

# Draw the signal graph
axis[0].set_title("the signal")
axis[0].set_xlabel("time")
axis[0].plot(timepoints, signal)
axis[0].grid()

# Draw the fft in real part
axis[1].set_title("the real part")
axis[1].set_xlabel("frequency")
real_part = fft.real
freq = np.fft.fftfreq(num_samples)
print(freq)
axis[1].plot(freq, fft.real)
axis[1].grid()

# Draw the fft_magnitude
axis[2].set_title("fft_magnitude")
axis[2].set_xlabel("frequency")
freq = np.fft.fftfreq(num_samples)
axis[2].plot(freq, fft_magnitude)
axis[2].grid()
 
plt.savefig("savefig_default2.png")
 
