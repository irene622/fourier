import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy import interpolate

begintime = -np.pi
endtime = np.pi
timelength = endtime - begintime
N = 512 # number of sample
l = 4
T = 2 ** l # period
sampling_freq = timelength / T
P = int(N / T) # number of period
timepoints = np.arange(begintime*P, endtime*P, sampling_freq)
# timepoints = np.arange(begintime, endtime*32, sampling_freq)
delta_omega = 2*np.pi / T
R = 2 # degree of phase function of lanmda_T

if len(timepoints) != N :
    # raise Exception("Be same the num of samples {} and timepoints {}.".format(N, len(timepoints)))
    print("test")

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
normalize_fft = fft / fft_magnitude
# fft_magnitude = fft_magnitude[:int(N/2)] # 0.007853076233206524, 3.8579644884021937

T = N / sampling_freq
k = np.arange(N)
freq = k / T
freq_list = [i for i in freq]

# filter the magnitude <= 10**-2
# data = []
# for i in range(int(N/2)) :
#     data.append((freq[i], fft_magnitude[i]))

# filtered_freq = []
# filtered_fft_magnitude = []
# for freq, fft_mag in data :
#     if fft_mag <= 10**(-2) :
#         filtered_freq.append(freq)
#         filtered_fft_magnitude.append(fft_mag)

f = interpolate.interp1d(freq_list[:int(N/2)], fft_magnitude[:int(N/2)], kind = 'quadratic')
# f = interpolate.interp1d(freq_list[:len(timepoints)], fft_magnitude, kind = 'quadratic')
h = 10**(-3)
Lambda_f = [] #the group delay of the desired signal
for freq in freq_list :
    try :
        value = f(freq + h) - f(freq)
        Lambda_f.append(value / h)
    except :
        Lambda_f.append(0)

# data = []
# for i in range(len(freq_list)) :
#     data.append((freq_list[i], fft_magnitude[i]))

phase = [cmath.phase(v) for v in normalize_fft]
g = interpolate.interp1d(freq_list, phase, kind = 'quadratic')

# calculate forward difference operater
h = 10**(-5)
Lambda_g = [] #the group delay of the desired signal
for freq in freq_list :
    try :
        value = g(freq + h) - g(freq)
        Lambda_g.append(abs(value / h))
    except :
        Lambda_g.append(0)


# Draw the group delay of the desired signal
plt.figure(figsize=(10,5))
plt.title("g")

# plt.plot(freq_list[:int(N/2)], Lambda_f[:int(N/2)])
# plt.plot(freq_list, Lambda_f[:(len(timepoints))])
plt.plot(freq_list[:int(N/2)], Lambda_g[:int(N/2)])
# plt.legend()
plt.grid()

plt.savefig("04group_delay_signal.png")


# _Lambda_f = []
# for i in range(int(N/2)) :
#     freq, fft_mag = data[i]
#     _Lambda_f.append(f(freq))
# Lambda_f = np.array(_Lambda_f)


# make matrix B
def Pi(n, k, T) :
    if -0.5 <= (n - T*k) / T < 0.5 :
        return 1
    else :
        return 0 

def b_nr(n, r) :
    b_nr = 0
    _P = int(P/2)
    for k in range(-_P, _P) :
        b_nr += (n - k*T) ** (2*r) * Pi(n, k, T)
    return b_nr

B_qT = np.zeros((N, int(R/2 + 1)))
for n in range(N) :
    for r in range(int(R/2 + 1)) :
        B_qT[n][r] = b_nr(int(n/2 + T/2), r)


B_q2m = np.zeros((N, int(R/2 + 1)))
for n in range(N) :
    for r in range(int(R/2 + 1)) :
        B_q2m[n][r] = b_nr(n/2, r)

D_psi = B_qT + B_q2m

# calculate Lambda_psi
conju_Dpsi = np.conjugate(D_psi)
c_hat1 = np.matmul(np.transpose(conju_Dpsi), conju_Dpsi)
c_hat2 = np.matmul(np.linalg.inv(c_hat1), np.transpose(conju_Dpsi))
c_hat = np.matmul(c_hat2, np.array(Lambda_g))

Lambda_psi = np.matmul(D_psi, c_hat)

plt.figure(figsize=(7,5))
plt.title("Lambda_psi")

plt.plot([i for i in range(Lambda_psi.shape[0])], Lambda_psi)
plt.grid()

plt.savefig("04group_delay_matched_wavelet.png")