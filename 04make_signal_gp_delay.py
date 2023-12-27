import numpy as np
import cmath
import matplotlib.pyplot as plt
from scipy import interpolate
from matching_wavelet import *

begintime = 0
endtime = 16
N = 512 # number of sample
l = 4
T = 2 ** l # period
delta_omega = 2*np.pi / T
sampling_freq = 1 / T
timepoints = np.arange(begintime, endtime, (endtime - begintime) / N)
P = int(N / T) # number of period
R = 2 # degree of phase function of lanmda_T

if len(timepoints) != N :
    raise Exception("Be same the num of samples {} and timepoints {}.".format(N, len(timepoints)))

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
# fft = np.fft.fftshift(fft)
fft_magnitude = abs(fft)
normalize_fft = fft / fft_magnitude

k = np.arange(len(normalize_fft))
freq = k / T
freq_list = [i for i in freq]

phase = [cmath.phase(v) for v in normalize_fft]
f = interpolate.interp1d(freq_list, phase, kind = 'quadratic')

# calculate forward difference operater
h = 10**(-2)
Lambda_f = [] #the group delay of the desired signal
for freq in freq_list :
    try :
        value = f(freq + h) - f(freq)
        Lambda_f.append(abs(value / h))
    except :
        Lambda_f.append(f(freq))


# Draw the group delay of the desired signal
plt.figure(figsize=(10,5))
plt.title("group delay of desired signal")

plt.plot(freq_list, Lambda_f, label="group delay of desired signal")
plt.legend()
plt.grid()

plt.savefig("04group_delay_signal.png")
Lambda_f = np.array(Lambda_f)

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
for n in range(N) : # row index
    for r in range(int(R/2 + 1)) : # column index
        B_qT[n][r] = b_nr(int(n/2 + T/2), r)


B_q2m = np.zeros((N, int(R/2 + 1)))
for n in range(N) : # row index
    for r in range(int(R/2 + 1)) : # column index
        B_q2m[n][r] = b_nr(n/2, r)

D_psi = B_qT + B_q2m

start_n = int(2**l / 3) + 1
end_n = int((2**(l+2)) / 3)

Y, _ = matching_wavelet(signal, N, l, T, delta_omega, P)
_Y = np.zeros(Lambda_f.shape)
_Y[start_n:end_n+1] = Y
Omega = _Y / sum(_Y)
bar_Lambda_f = np.multiply(Lambda_f, Omega) # elementary wise product

# calculate Lambda_psi
_Omega = np.zeros(D_psi.shape)
for col in range(_Omega.shape[1]) :
    _Omega[:,col] = Omega
bar_Dpsi = np.multiply(D_psi, _Omega)
c_hat1 = np.matmul(np.transpose(bar_Dpsi), bar_Dpsi)
c_hat2 = np.matmul(np.linalg.inv(c_hat1), np.transpose(bar_Dpsi))
c_hat = np.matmul(c_hat2, np.array(bar_Lambda_f))

Lambda_psi = np.matmul(D_psi, c_hat)

plt.figure(figsize=(7,5))
plt.title("Lambda_psi")

plt.plot([i for i in range(Lambda_psi.shape[0])], Lambda_psi)
plt.grid()

plt.savefig("04group_delay_matched_wavelet.png")