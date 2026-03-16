# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import matplotlib.pyplot as plt
import numpy as np

from scipy import special as sp

def q_function(snr):
    value = np.sqrt(pow(10, snr / 10) / 5)
    return 0.5 - 0.5*sp.erf(value/np.sqrt(2))

def db_to_val(snr):
    # convert dB to linear
    return 10 ** (snr / 10)

def P_e(snr, M, gamma):
    N_e = 4 * (1 - 1/(np.sqrt(M)))
    return N_e * q_function(np.sqrt((3 * snr * gamma) / (M-1)))

M = 4
gamma = 5
snr = 10
snrs = [i * 2 for i in range(0, 16)]



plt.semilogy(snrs, [P_e(db_to_val(snr), M, gamma) for snr in snrs], label = "Expected $P_e$", color = 'red', ls = 'solid')
plt.title(f"SNR (dB) vs. $P_e$ for {M}-QAM with Coding Gain of {gamma}")
plt.xlabel("SNR (dB)")
plt.ylabel("BER")
plt.legend()
plt.show()

