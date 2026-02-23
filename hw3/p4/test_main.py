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
    return np.sum(np.bitwise_count(arr1 - arr2)) / (arr1.size * bits_per_symbol)

PREAMBLE = np.array((1, 0, 1, 0, 1, 1, 1, 1) * 4)
bits_per_symbol = 1
symbol_size = 1
snr = 100  # in dB
sample_size = int(1e6) // symbol_size

# get image
test_input = image_to_bits('./photos/monalisa_diff.png')
# print("Input image:", test_input)

# transform image
transmitter_output = transmitter(test_input, preamble=PREAMBLE, symbol_size=symbol_size, bits_per_symbol=bits_per_symbol)
# print("Transformer output:", transmitter_output)
# add zeros before and after data
r = round(np.random.rand() * 6e5)
signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(6e5-r))))
# print("Padded signal:", signal)

# make sure it's the right length
assert len(signal) == int(1e6), f"{len(signal)}"

# add noise
noisy_output = truncate_add_noise_passband(signal, snr)
noisy_output = signal
print("Noisy signal:", noisy_output)

# receive signal
received_signal, index = receiver(noisy_output, preamble=PREAMBLE, bits_per_symbol=bits_per_symbol, symbol_size=symbol_size)

print(index, " ", r)
if index != r:
    print("Preamble not identified correctly")
if received_signal is None:
    print("Unable to display image")
    
# print("Received signal:", received_signal)

plt.imshow(test_input, cmap='gray', vmin=0, vmax=1)
plt.title("Original Signal")
plt.show()

if received_signal is not None:
    plt.imshow(received_signal, cmap='gray', vmin=0, vmax=1)
    # plt.title("Received Signal")
    plt.show()
    print("Error rate:", calculate_error_rate(test_input, received_signal))

snrs = range(-10, 21, 1)
det_rates = []
dec_rates = []
for test_snr in snrs:
    print(f"Starting with SNR = {test_snr}")
    detected = 100
    decoded = 100
    for _ in range(100):
        # add zeros before and after data
        r = round(np.random.rand() * 9e5)
        signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(9e5-r))))
        noisy_output = truncate_add_noise_passband(signal, test_snr)
        received_signal, index = receiver(noisy_output, preamble=PREAMBLE, bits_per_symbol=bits_per_symbol)
        if index != r:
            print("index wrong")
            print(index, r)
            detected -= 1
            decoded -= 1
            continue
        if received_signal is None:
            decoded -= 1
            continue
        if test_input.shape != received_signal.shape:
            decoded -= 1
            continue
        if calculate_error_rate(test_input, received_signal) != 0:
            decoded -= 1
    det_rates.append(detected / 100)
    dec_rates.append(decoded / 100)

plt.title(f"SNR vs. Error Rates")
plt.xlabel("SNR (dB)")
plt.ylabel("Rate")
plt.plot(snrs, det_rates, label="Detecting Rate")
plt.plot(snrs, dec_rates, label="Decoding Rate")
plt.legend()
plt.show()

"""
iii) The difference is about 8dB.  This exists because if you do not decode the preamble correctly, you cannot decode the rest of the message correcly.
No necessarily, there will be a difference.  Addiitionally, if you can get the preamble correct, you are still not garunteed to get the rest of the message correct.
The preamble only represents a very small part of the bits that could possibly have errors.
"""
