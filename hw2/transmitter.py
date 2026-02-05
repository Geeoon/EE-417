# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

def to_bits(val: int, out_len: int=8) -> np.ndarray:
    assert val < 2 ** out_len, "output length isn't large enough to hold all bits"
    assert val >= 0, "can't convert negative numbers"
    out = np.array([])
    while val:
        out = np.append(out, val % 2)
        val >>= 1
    return np.concatenate([[0] * (out_len - len(out)), out]).astype(np.uint8)

def transmitter(input_signal: np.ndarray, symbol_size: int=10, bits_per_symbol: int=1, amplitude: float=2.0) -> np.ndarray:
    """
    Converts a series of bits for transmission.

    :param input_signal: the bits to transmit 
    :type input_signal: np.ndarray
    :param symbol_size: the number of samples per symbol
    :type symbol_size: int
    :param bits_per_symbol: the number of bits per symbol
    :type bits_per_symbol: int
    :param amplitude: the amplitude of the output.  Goes from [-amplitde/2 to ampltidue/2]
    :type amplitude: float
    
    :return: the transformed signal
    :rtype: np.ndarray
    """
    assert(symbol_size >= 1)
    assert(bits_per_symbol >= 1)
    assert(amplitude >= 1)
    assert(np.max(input_signal) < 2 ** bits_per_symbol)
    input_signal = np.concatenate([to_bits(len(input_signal)), input_signal])  # add length to front of message
    input_signal = np.concatenate([(1, 0, 1, 0, 1, 1, 1, 1), input_signal])  # add preamble
    
    out = (input_signal * (amplitude / ((2 ** bits_per_symbol) - 1))) - amplitude / 2
    out = np.repeat(out, symbol_size)
    return out

out = transmitter([1, 0, 1, 1], symbol_size=1)
print(out)
