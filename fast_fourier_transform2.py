import numpy as np
import matplotlib.pyplot as plt
 
fs = 2 
t = np.arange(-np.pi, np.pi, 1 / fs) # from 0 to 3, each interval 1/100. len(t) = (3-0)*fs
print(f"len(t): {len(t)}") # 샘플의 갯수

signal = t - t + 1
 
# Fourier transform
fft = np.fft.fft(signal) / len(signal)  #  반환값을 '양의 영역 다음에 음의 영역 순서'로 반환한다.
print(f"fft: {fft[:2]}")
 
fft_magnitude = abs(fft)


# Draw signal graph
plt.subplot(2,1,1)
plt.plot(t,signal)
plt.grid()

# Draw fft_magnitude graph using stem graph.
plt.subplot(2,1,2)
length = len(signal)
f = np.linspace(-(fs / 2), fs / 2, length) 
plt.stem(f, np.fft.fftshift(fft_magnitude)) # 줄기와 잎 그림.
plt.ylim(0,2.5)
plt.grid()
 
plt.savefig('savefig_default2.png')
 
