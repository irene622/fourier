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
