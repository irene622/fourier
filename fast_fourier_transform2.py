import numpy as np
import matplotlib.pyplot as plt
 
fs = 2 
# np.arange(a, b, c) : a부터 b까지 범위를 c간격으로 뽑는다.
t = np.arange(-np.pi, np.pi, 1 / fs)
print(f"len(t): {len(t)}") # 샘플의 갯수

# signal 정의
signal = 1
 
# Discrete Fourier transform with Fast Fourier Transform
fft = np.fft.fft(signal) / len(signal)  #  반환값을 '양의 영역 다음에 음의 영역 순서'로 반환한다.
print(f"fft: {fft[:2]}")
 
fft_magnitude = abs(fft)


# Draw the signal graph
plt.subplot(23,1,1)
plt.plot(t,signal)
plt.grid()

# Draw the fft in real part
plt.subplot(3,1,2)
plt.plot(t,signal.real)
plt.grid()

# Draw fft_magnitude graph using stem graph.
plt.subplot(3,1,3)
length = len(t)
f = np.linspace(-(fs / 2), fs / 2, length) 
plt.stem(f, np.fft.fftshift(fft_magnitude)) # 줄기와 잎 그림.
plt.ylim(0,2.5)
plt.grid()
 
plt.savefig('savefig_default2.png')
 
