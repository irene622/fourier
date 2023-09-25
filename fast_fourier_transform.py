import numpy as np
import matplotlib.pyplot as plt
 
fs = 1 
t = np.arange(0, 3, 1 / fs) # from 0 to 3, each interval 1/100. len(t) = (3-0)*fs

f1 = 35
f2 = 10
# 35Hz를 갖는 0.6 진폭의 신호와 10Hz를 갖는 3 진폭 신호를 생성한 뒤,
# 두 신호를 더하여 설명에 사용할 신호를 생성한다.
signal = 0.6 * np.sin(2 * np.pi * f1 * t) + 3 * np.cos(2 * np.pi * f2 * t + np.pi/2)
 
# Fourier transform
fft = np.fft.fft(signal) / len(signal)  #  반환값을 '양의 영역 다음에 음의 영역 순서'로 반환한다.
 
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
 
plt.savefig('savefig_default.png')
 
