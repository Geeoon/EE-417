# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import matplotlib.pyplot as plt
import numpy as np

from receiver import receiver
from transmitter import transmitter
from truncate_add_noise_real import truncate_add_noise_real
from image_to_bits import image_to_bits

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
    assert(len(arr1) == len(arr2))
    return np.sum(np.bitwise_count(arr1 - arr2)) / (len(arr1) * bits_per_symbol)


bits_per_symbol = 1
symbol_size = 1
snr = 20  # in dB
sample_size = int(1e6) // symbol_size

# get image
test_input = image_to_bits('./photos/monalisa_diff.png')
print("Input image:", test_input)

# transform image
transmitter_output = transmitter(test_input, symbol_size=symbol_size, bits_per_symbol=bits_per_symbol)
print("Transformer output:", transmitter_output)
# add zeros before and after data
r = round(np.random.rand() * 9e5)
signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(9e5-r))))
print("Padded signal:", signal)

# make sure it's the right length
assert len(signal) == int(1e6)

# add noise
noisy_output = truncate_add_noise_real(signal, snr)
print("Noisy signal:", noisy_output)

# receive signal
received_signal = receiver(noisy_output, bits_per_symbol=bits_per_symbol)
print("Received signal:", received_signal)

quit()

# interpret signal
# average the samples for each symbol
# final_output = np.rint(received_signal.reshape(-1, symbol_size).mean(axis=1)).astype(np.int64)
# print("Final output:", final_output)
# print("Bit error rate:", calculate_error_rate(test_input, final_output))


# EXTRA CREDIT
# part i
# snrs = range(-10, 31, 1)
# errs = []
# bits_per_symbol = 1
# symbol_size = 10
# sample_size = int(1e6) // symbol_size
# for test_snr in snrs:
#     snr = test_snr
#     test_input = rng.integers(low=0, high=2**bits_per_symbol, size=sample_size)
#     transmitter_output = transmitter(test_input, bits_per_symbol=bits_per_symbol, symbol_size=symbol_size)
#     noisy_output = truncate_add_noise_real(transmitter_output, snr)
#     received_signal = receiver(noisy_output, bits_per_symbol=bits_per_symbol)
#     final_output = np.rint(received_signal.reshape(-1, symbol_size).mean(axis=1)).astype(np.int64)
#     errs.append(calculate_error_rate(test_input, final_output))
# plt.title(f"SNR vs. BER (Symbol Size = {symbol_size} Samples, Bits Per Symbol = {bits_per_symbol})")
# plt.xlabel("SNR (dB)")
# plt.ylabel("Bit Error Rate (BER)")
# plt.plot(snrs, errs)
# plt.show()

# part ii
# snrs = range(-10, 31, 1)
# errs = []
# bits_per_symbol = 1
# symbol_size = 5
# sample_size = int(1e6) // symbol_size
# for test_snr in snrs:
#     snr = test_snr
#     test_input = rng.integers(low=0, high=2**bits_per_symbol, size=sample_size)
#     transmitter_output = transmitter(test_input, bits_per_symbol=bits_per_symbol, symbol_size=symbol_size)
#     noisy_output = truncate_add_noise_real(transmitter_output, snr)
#     received_signal = receiver(noisy_output, bits_per_symbol=bits_per_symbol)
#     final_output = np.rint(received_signal.reshape(-1, symbol_size).mean(axis=1)).astype(np.int64)
#     errs.append(calculate_error_rate(test_input, final_output))
# plt.title(f"SNR vs. BER (Symbol Size = {symbol_size} Samples, Bits Per Symbol = {bits_per_symbol})")
# plt.xlabel("SNR (dB)")
# plt.ylabel("Bit Error Rate (BER)")
# plt.plot(snrs, errs)
# plt.show()

# part iii
# snrs = range(-10, 31, 1)
# errs = []
# bits_per_symbol = 2
# symbol_size = 10
# sample_size = int(1e6) // symbol_size
# for test_snr in snrs:
#     snr = test_snr
#     test_input = rng.integers(low=0, high=2**bits_per_symbol, size=sample_size)
#     transmitter_output = transmitter(test_input, bits_per_symbol=bits_per_symbol, symbol_size=symbol_size)
#     noisy_output = truncate_add_noise_real(transmitter_output, snr)
#     received_signal = receiver(noisy_output, bits_per_symbol=bits_per_symbol)
#     final_output = np.rint(received_signal.reshape(-1, symbol_size).mean(axis=1)).astype(np.int64)
#     errs.append(calculate_error_rate(test_input, final_output, bits_per_symbol=bits_per_symbol))
# plt.title(f"SNR vs. BER (Symbol Size = {symbol_size} Samples, Bits Per Symbol = {bits_per_symbol})")
# plt.xlabel("SNR (dB)")
# plt.ylabel("Bit Error Rate (BER)")
# plt.plot(snrs, errs)
# plt.show()

