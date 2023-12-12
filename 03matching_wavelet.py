import numpy as np
import matplotlib.pyplot as plt

begintime = 0
endtime = 8*np.pi
timelength = endtime - begintime
N = 512 # number of sample
sampling_interval = timelength / N
timepoints = np.arange(begintime, endtime, sampling_interval)
l = 4
T = 2 ** l # period
delta_omega = 2*np.pi / T
P = int(N / T) # number of period
R = 2 # degree of phase function of lanmda_T


alpha = 2.0
f_0 = 0.8
def f_T(x) :
    if x < 0 :
        u = 0
    else :
        u = 1 
    return x * np.exp(-alpha * x) * np.cos(2*np.pi*f_0*x) * u
signal = [f_T(x) for x in timepoints]

def make_A(size) :
    # Make matrix A
    A = np.zeros(size)
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
    return A

def matching_wavelet(signal, N, l, T, delta_omega, P) :
    # Make matrix A
    A = make_A((11,16))

    # Make matrix W
    fft = np.fft.fft(signal)
    # fft = np.fft.fftshift(fft)

    # size = fft.shape[0]
    # _permuatation_mat = np.identity(size)
    # permuatation_mat = np.zeros((size, size))
    # n = 0
    # for idx in range(0, size, 2) :
    #     permuatation_mat[n, :] = _permuatation_mat[idx, :].copy()
    #     n += 1
    # for idx in range(1, size, 2) :
    #     permuatation_mat[n] = _permuatation_mat[idx]
    #     n += 1
    # fft = np.matmul(fft, permuatation_mat)
    
    fft_magnitude = abs(fft) ** 2

    start_n = int(2**l / 3) + 1
    end_n = int((2**(l+2)) / 3)

    # make matrix W
    W = np.array(fft_magnitude[start_n: end_n+1]) # W.shape = (end_n - start_n + 1, 1)
    W = W * 1/T # 이게 뭐야...

    if end_n - start_n + 1 == W.shape[0] :
        print("Correspond the size of W")
        print(f"W.shape: {W.shape} | start_n: {start_n} | end_n: {end_n}")
    else :
        print(f"W.shape have to equal to {end_n - start_n + 1}. Configure the size.")

    # Calculate a
    A_t = np.transpose(A)
    numerate1 = np.transpose(np.ones(A.shape[0])) # 1.tr
    numerate2 = np.linalg.inv(np.matmul(A, A_t))
    numerate3 = np.matmul(np.matmul(np.matmul(numerate1, numerate2), A), W) # 분자
    denominator1 = np.matmul(numerate1, numerate2)
    denominator2 = np.matmul(denominator1, np.ones(A.shape[0])) # 분모
    a = numerate3 / denominator2
    print(f"a: {a}") # a

    # Calculate Y
    Y1 = np.linalg.inv(np.matmul(A, A_t))
    Y2 = np.matmul(A_t, Y1)
    Y3 = np.ones(A.shape[0]) - 1/a * np.matmul(A, W)
    Y = 1/a * W + np.matmul(Y2, Y3)
    # Y = Y * 1/sum(Y)
    return Y, W

if __name__ == "__main__" :
    # Matching Discrete Spectrum Amplitude
    Y, W = matching_wavelet(signal, N, l, T, delta_omega, P)

    # Draw the vector Y
    plt.title("Amplitude matching")
    plt.plot([i for i in range(Y.shape[0])], Y, label='Amplitude Matching')
    plt.plot([i for i in range(Y.shape[0])], W, '--', label='fft magnitude of singal')
    plt.grid()
    plt.legend()

    plt.savefig("03amplitude_matching_wavelet.png")