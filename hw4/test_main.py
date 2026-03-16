# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import matplotlib.pyplot as plt
import numpy as np

from scipy import special as sp

from image_to_bits import image_to_bits
from transmitter import transmitter
from truncate_add_noise_passband import truncate_add_noise_passband
from receiver import receiver

# thanks stack overflow for this one
def q_function(snr):
    value = np.sqrt(pow(10, snr / 10) / 5)
    return 0.5 - 0.5*sp.erf(value/np.sqrt(2))
    
def calculate_error_rate(arr1: np.ndarray, arr2: np.ndarray, bits_per_symbol: int=1) -> float:
    """
    Calculates the proportion of elements that are not the same in both input arrays
    
    :param truth: the first array to compare
    :type truth: np.ndarray
    :param test: the second array to compare
    :type test: np.ndarray
    :param bits_per_symbol: the number of bits per symbol
    :type bits_per_symbol: int

    :return: the bit error rate [0, 1]
    :rtype: float
    """
    assert(bits_per_symbol > 0)
    assert(arr1.shape == arr2.shape)
    diff = arr1 - arr2
    reshaped = diff.reshape(-1, 4)  # 16-QAM has 4 bits
    bit_err = np.sum(np.bitwise_count(diff)) / (arr1.size * bits_per_symbol)
    sym_err = np.sum(np.any(reshaped != 0, axis=1)) / len(reshaped)
    return bit_err, sym_err

def db_to_val(snr):
    # convert dB to linear
    return 10 ** (snr / 10)

def P_e(snr, M, gamma):
    N_e = 4 * (1 - 1/(np.sqrt(M)))
    return N_e * q_function(np.sqrt((3 * snr * gamma) / (M-1)))

M = 4
gamma = 5
PREAMBLE = np.array([1, 0, 1, 0, 1, 1, 1, 1]*8)
bits_per_symbol = 1
symbol_size = 3
snr = 20  # in dB

# get image
test_input = image_to_bits('./photos/monalisa_diff.png', 300)
print("Input image:", test_input)
print("input image shape: ", np.shape(test_input))

# transform image
transmitter_output = transmitter(test_input, preamble=PREAMBLE, symbol_size=symbol_size, bits_per_symbol=bits_per_symbol)
# print("Transmitter output:", transmitter_output)

if True:
    # add zeros before and after data
    r = round(np.random.rand() * (1e6 - len(transmitter_output)))
    signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(1e6 - len(transmitter_output) - r))))

    # print("Padded signal:", signal)

    # make sure it's the right length
    assert len(signal) == int(1e6), f"{len(signal)}"

    # add noise
    noisy_output = truncate_add_noise_passband(signal, snr)
    # noisy_output = signal  # skip noise for testing
    # print("Noisy signal:", noisy_output)

    print("preamble at: ", r)

    # receive signal
    received_hard, received_soft, index, hard_dims, soft_dims = receiver(noisy_output, preamble=PREAMBLE, expected_preamble_idx=r)

    if index != r:
        print("Preamble not identified correctly")

    # print("hard decoded image shape: ", np.shape(received_hard))
    # print("soft decoded image shape: ", np.shape(received_soft))

    plt.title("original image")
    plt.imshow(test_input, cmap='gray', vmin=0, vmax=1)
    plt.show()

    if received_hard is None:
        print("Unable to display hard-decoded image")
    else:
        plt.title("hard decoding image")
        plt.imshow(np.reshape(received_hard[:-2], hard_dims), cmap='gray', vmin=0, vmax=1)
        plt.show()

    if received_soft is None:
        print("Unable to display soft-decoded image")
    else:
        plt.title("soft decoding image")
        plt.imshow(np.reshape(received_soft[:-2], soft_dims), cmap='gray', vmin=0, vmax=1)
        plt.show()

print("\n### NOW CALCULATING BER ###\n")

snrs = [i * 2 for i in range(0, 16)]
hard_ber = [0] * 16
hard_pe  = [0] * 16

soft_ber = [0] * 16
soft_pe  = [0] * 16

flat_input = np.ravel(test_input)
img_length = len(flat_input)

half_len = img_length / 2
flat_half  = np.split(flat_input, half_len)

# simulation
for snr in snrs:
    trials = 0
    hard_errors_collected = 0
    soft_errors_collected = 0
    curr = snr // 2
    while hard_errors_collected < 50 and soft_errors_collected < 50:
        trials += 1
        noisy_signal = truncate_add_noise_passband(transmitter_output, snr, len(transmitter_output))
        received_hard, received_soft, _, _, _ = receiver(noisy_signal, preamble = PREAMBLE, expected_preamble_idx = 0)

        received_hard = received_hard[:-2]
        received_soft = received_soft[:-2]

        # automatically count missed inputs as errors
        if len(received_hard) != len(flat_input):
            received_hard = np.concatenate([received_hard, flat_input[len(received_hard):] ^ 1])
        if len(received_soft) != len(flat_input):
            received_soft = np.concatenate([received_soft, flat_input[len(received_soft):] ^ 1])

        hard_bit_error_count = np.sum(flat_input != received_hard)
        soft_bit_error_count = np.sum(flat_input != received_soft)

        hard_errors_collected += hard_bit_error_count
        soft_errors_collected += soft_bit_error_count

        hard_ber[curr] += hard_bit_error_count / img_length
        soft_ber[curr] += soft_bit_error_count / img_length

        hard_pe[curr] += np.sum(np.any(np.array(flat_half) != np.array(np.split(received_hard, half_len)), axis=1)) / half_len
        soft_pe[curr] += np.sum(np.any(np.array(flat_half) != np.array(np.split(received_soft, half_len)), axis=1)) / half_len

    hard_ber[curr] /= trials
    soft_ber[curr] /= trials
    hard_pe[curr] /= trials
    soft_pe[curr] /= trials

    print("SNR =", snr)
    print("Hard BER =", hard_ber[curr])
    print("Soft BER =", soft_ber[curr])
    print("Hard PE =", hard_pe[curr])
    print("Soft PE =", soft_pe[curr])

plt.semilogy(snrs, hard_ber, label = "Hard Decoding BER", color = 'red', ls = 'solid')
plt.semilogy(snrs, soft_ber, label = "Soft Decoding BER", color = 'green', ls = 'solid')
plt.semilogy(snrs, hard_pe, label = "Hard Decoding $P_e$", color = 'orange', ls = 'dashed')
plt.semilogy(snrs, soft_pe, label = "Soft Decoding $P_e$", color = 'blue', ls = 'dashed')
plt.title("Hard vs. Soft Decoding Error Rates @ SNR between 0 and 30 dB")
plt.xlabel("SNR (dB)")
plt.ylabel("Error Rate")
plt.legend()
plt.show()

# expected P_e
plt.semilogy(snrs, [P_e(db_to_val(snr), M, gamma) for snr in snrs], label = "Expected $P_e$", color = 'red', ls = 'solid')
plt.title(f"SNR (dB) vs. $P_e$ for {M}-QAM with Coding Gain of {gamma}")
plt.xlabel("SNR (dB)")
plt.ylabel("Error Rate")
plt.legend()
plt.show()

"""
# Quantitative Analysis
The empirical results we got from the simulation are different from the expected results for 4-QAM with a coding gain of 5 (derived from our G matrix).  Although this they differ, it is to be expected.  The expected results are just the upper bound on the error rates.  So our empirical results being lower makes sense.

Between the soft and hard decoding, we can see that soft decoding has much lower error rate the hard decoding.  This also makes sense because soft decoding takes into account the actual symbols rather than collapsing the information into two bits.

Between hard and soft decoding, we observe an SNR difference of about dB.
"""