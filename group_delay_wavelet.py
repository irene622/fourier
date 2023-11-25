import numpy as np
import matplotlib.pyplot as plt
from matching_wavelet import *

begintime = 0
endtime = 10
timelength = endtime - begintime
samplingfreq = 512
samplinginterval = timelength / samplingfreq
timepoints = np.arange(begintime, endtime, samplinginterval)
N = samplingfreq # number of sample
l = 4
T = 2 ** l # period
delta_omega = 2*np.pi / T
P = int(N / T) # number of period
R = 2 # degree of phase function of lanmda_T
a = 1.0348 # constant. hyperparameters

alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x <= 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(x) for x in timepoints]


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
        b_nr += (n - k*T) ^ (2*r) * Pi(n, k, T)
    return b_nr

B_qT = np.zeros((N, int(R/2 + 1)))
for n in range(N) :
    for r in range(int(R/2 + 1)) :
        B_qT[n][r] = b_nr(int(n/2 + T/2), r)
print(B_qT.shape)
print(B_qT[256:260])