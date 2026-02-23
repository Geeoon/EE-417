# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import matplotlib.pyplot as plt
import numpy as np

from receiver import receiver
from transmitter import transmitter
from truncate_add_noise_passband import truncate_add_noise_passband
from image_to_bits import image_to_bits
from symbol_detector_to_bits import symbol_detector_to_bits

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

PREAMBLE = np.array((1, 0, 1, 0, 1, 1, 1, 1) * 4)
bits_per_symbol = 1
symbol_size = 3
snr = 100  # in dB

# get image
test_input = image_to_bits('./photos/monalisa_diff.png')
print("Input image:", test_input)

# transform image
transmitter_output = transmitter(test_input, preamble=PREAMBLE, symbol_size=symbol_size, bits_per_symbol=bits_per_symbol)
print("Transformer output:", transmitter_output)
# add zeros before and after data
r = round(np.random.rand() * 6e5)
signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(6e5-r))))
print("Padded signal:", signal)

# make sure it's the right length
assert len(signal) == int(1e6), f"{len(signal)}"

# add noise
noisy_output = truncate_add_noise_passband(signal, snr)
print("Noisy signal:", noisy_output)

# receive signal
received_signal, index = receiver(noisy_output, preamble=PREAMBLE, bits_per_symbol=bits_per_symbol, symbol_size=symbol_size)

if index != r:
    print("Preamble not identified correctly")
if received_signal is None:
    print("Unable to display image")
    
print("Received signal:", received_signal)

plt.imshow(test_input, cmap='gray', vmin=0, vmax=1)
plt.show()

if received_signal is not None:
    plt.imshow(received_signal, cmap='gray', vmin=0, vmax=1)
    plt.show()

N = 3
snrs = range(0, 21, 2)
det_rates = []
dec_rates = []
sers = []
bers = []
for test_snr in snrs:
    print(f"Starting with SNR = {test_snr}")
    detected = N
    decoded = N
    ber = 0
    ser = 0
    for _ in range(N):
        # add zeros before and after data
        r = round(np.random.rand() * 6e5)
        signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(6e5-r))))
        noisy_output = truncate_add_noise_passband(signal, test_snr)
        received_signal, index = receiver(noisy_output, preamble=PREAMBLE, bits_per_symbol=bits_per_symbol, symbol_size=symbol_size)
        if index != r:
            print("index wrong")
            print(index, r)
            detected -= 1
            decoded -= 1
            ber += 1
            ser += 1
            continue
        if received_signal is None:
            decoded -= 1
            ber += 1
            ser += 1
            continue
        if test_input.shape != received_signal.shape:
            decoded -= 1
            ber += 1
            ser += 1
            continue
        this_ber, this_ser = calculate_error_rate(test_input, received_signal)
        print(this_ber, this_ser)
        if this_ber != 0:
            decoded -= 1
        ber += this_ber
        ser += this_ser
    det_rates.append(detected / N)
    dec_rates.append(decoded / N)
    bers.append(ber / N)
    sers.append(ser / N)

plt.title(f"SNR vs. Error Rates")
plt.xlabel("SNR (dB)")
plt.ylabel("Rate")
plt.semilogy(snrs, det_rates, label="Detecting Rate")
plt.semilogy(snrs, dec_rates, label="Decoding Rate")
plt.legend()
plt.show()

plt.title(f"SNR vs. Error Rates")
plt.xlabel("SNR (dB)")
plt.ylabel("Rate")
plt.semilogy(snrs, bers, label="Bit Error Rate")
plt.semilogy(snrs, sers, label="Symbol Error Rate")
plt.legend()
plt.show()
