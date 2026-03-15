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

PREAMBLE = np.array([1, 0, 1, 0, 1, 1, 1, 1])
bits_per_symbol = 1
symbol_size = 3
snr = 10  # in dB

# get image
test_input = image_to_bits('./photos/test_checker.png', 32) # to test, call this with 32 as a second parameter
print("Input image:", test_input)
print("input image shape: ", np.shape(test_input))

# transform image
transmitter_output = transmitter(test_input, preamble=PREAMBLE, symbol_size=symbol_size, bits_per_symbol=bits_per_symbol)
# print("Transmitter output:", transmitter_output)

# add zeros before and after data
r = round(np.random.rand() * (1e6 - len(transmitter_output))) # to test, do not add this padding
signal = np.concatenate((np.zeros(r), transmitter_output, np.zeros(int(1e6 - len(transmitter_output) - r))))

# print("Padded signal:", signal)

# make sure it's the right length
assert len(signal) == int(1e6), f"{len(signal)}"

# add noise
noisy_output = truncate_add_noise_passband(signal, snr) # to test, use 2182 in the last parameter
# noisy_output = signal  # skip noise for testing
# print("Noisy signal:", noisy_output)

print("preamble at: ", r)

# receive signal
received_hard, received_soft, index = receiver(noisy_output, preamble=PREAMBLE, expected_preamble_idx=r)

if index != r:
    print("Preamble not identified correctly")

print("hard decoded image shape: ", np.shape(received_hard))
print("soft decoded image shape: ", np.shape(received_soft))

plt.imshow(test_input, cmap='gray', vmin=0, vmax=1)
plt.show()

if received_hard is None:
    print("Unable to display hard-decoded image")
else:
    plt.imshow(received_hard, cmap='gray', vmin=0, vmax=1)
    plt.show()

if received_soft is None:
    print("Unable to display soft-decoded image")
else:
    plt.imshow(received_soft, cmap='gray', vmin=0, vmax=1)
    plt.show()
