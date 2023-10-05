import numpy as np
import matplotlib.pyplot as plt
 
begintime = -np.pi
endtime = np.pi
samplingfreq = 200
samplinginterval = 1 / samplingfreq
timepoints = np.arange(begintime, endtime, samplinginterval)
print(f"len(timepoints): {len(timepoints)}") # 샘플의 갯수

# create signal
signal = timepoints - timepoints + 1
 
# Discrete Fourier transform with Fast Fourier Transform
fft = np.fft.fft(signal) / len(timepoints) # normalize
print(f"fft: {fft[:10]}")
 
fft_magnitude = abs(fft)

# create subplot
figure, axis = plt.subplots(3, 1)
plt.subplots_adjust(hspace=1)

# Draw the signal graph
axis[0].set_title("the signal")
axis[0].set_xlabel('Time')
axis[0].plot(timepoints,signal)
axis[0].grid()

# Draw the fft in real part
axis[1].set_title("fft real part")
axis[1].set_xlabel("frequency")
freq = np.fft.fftfreq(timepoints.shape[-1])
axis[1].plot(freq,fft.real)
axis[1].grid()

# Draw fft_magnitude graph using stem graph.
length = len(timepoints)
f = np.linspace(-(samplingfreq / 2), samplingfreq / 2, length) 
axis[2].set_title("fft_magnitude")
axis[2].set_xlabel("frequency")
axis[2].stem(f, np.fft.fftshift(fft_magnitude)) # 줄기와 잎 그림.
axis[2].set_ylim(-0.5,0.5)
axis[2].grid()
 
plt.savefig('savefig_default2.png')
 
