import numpy as np
import matplotlib.pyplot as plt

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

print(len(timepoints))

alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x <= 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(x) for x in timepoints]

# Make matrix A
A = np.zeros((11,16))
A[0][0] = A[0][6] = 1
A[1][1] = A[1][8] = 1
A[2][2] = A[2][10] = 1
A[3][3] = A[3][12] = 1
A[4][4] = A[4][14] = 1
A[5][5] = A[5][15] = 1
A[6][6] = A[6][14] = 1
A[7][7] = A[7][13] = 1
A[8][8] = A[8][12] = 1
A[9][9] = A[9][11] = 1
A[10][10] = 2


# Make matrix W
fft = np.fft.fft(signal)
fft_magnitude = abs(fft)

start_n = int(2**l / 3) + 1
end_n = int((2**(l+2)) / 3)
W = np.array(fft_magnitude[start_n-1: end_n]) # W.shape = (end_n - start_n + 1, 1)
if end_n - start_n + 1 == W.shape[0] :
    print("Right the size of W")
    print(f"W.shape {W.shape} | end_n - start_n {end_n - start_n + 1}")
else :
    print(f"W.shape have to equal to {end_n - start_n + 1}. Configure the size.")

# Calculate Y
a = 1.0348 # constant. hyperparameters
A_t = np.transpose(A)
Y1 = np.matmul(A_t, np.linalg.inv(np.matmul(A, A_t)))
Y2 = np.ones(A.shape[0]) - 1/a * np.matmul(A, W)
Y = 1/a * W + np.matmul(Y1, Y2)
print(Y.shape)


# Draw the vector Y
plt.title("Amplitude matching")
plt.plot([i for i in range(Y.shape[0])], Y, label='Amplitude Matching')
plt.plot([i for i in range(Y.shape[0])], fft_magnitude[start_n-3:end_n-2], '--', label='Amplitude of singal fft')
plt.grid()
plt.legend()

plt.savefig("Amplitude_matching.png")