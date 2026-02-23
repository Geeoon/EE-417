# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

from bit_to_symbol import bit_to_symbol

def to_symbol(val: int, bits_per_symbol: int=1, out_len: int=16) -> np.ndarray:
    assert val < 2 ** (out_len * bits_per_symbol), "output length isn't large enough to hold all bits"
    assert val >= 0, "can't convert negative numbers"
    out = np.array([])
    while val:
        out = np.append(val % (2 ** bits_per_symbol), out)
        val >>= bits_per_symbol
    out = np.concatenate([[0] * (out_len - len(out)), out]).astype(np.uint8)
    return out

def transmitter(input_signal: np.ndarray, preamble: np.ndarray = (1, 0, 1, 0, 1, 1, 1, 1), symbol_size: int=1, bits_per_symbol: int=1) -> np.ndarray:
    """
    Converts a series of bits for transmission.

    :param input_signal: the bits to transmit 
    :type input_signal: np.ndarray
    :param symbol_size: the number of samples per symbol
    :type symbol_size: int
    :param bits_per_symbol: the number of bits per symbol
    :type bits_per_symbol: int
    
    :return: the transformed signal
    :rtype: np.ndarray
    """
    assert symbol_size >= 1
    assert bits_per_symbol >= 1
    assert np.max(input_signal) == 1 or np.max(input_signal) == 0  # make sure it's a 1 bit stream
    # add length and x, y dimensions to front of message
    print("X, before:", to_symbol(input_signal.shape[0], bits_per_symbol=bits_per_symbol))
    print("Y, before:", to_symbol(input_signal.shape[1], bits_per_symbol=bits_per_symbol))
    input_signal = np.concatenate([to_symbol(input_signal.shape[0], bits_per_symbol=bits_per_symbol),
                                   to_symbol(input_signal.shape[1], bits_per_symbol=bits_per_symbol),
                                   input_signal.flatten()])
    input_signal = np.concatenate([preamble, input_signal])  # add preamble
    assert len(input_signal) < int(4e5), "Image too large"
    out = bit_to_symbol(input_signal)
    out = np.repeat(out, symbol_size)
    
    # pad with zeros
    out = np.concatenate((out, np.zeros(int(4e5 - len(out)))))
    
    # make sure the output is the right size
    assert len(out) >= 4e5, f"{len(out)} < 4e5"
    
    """
    So the output looks as such
    bits start-0: preamble
    bits 0-15: x-dimension
    bits 16-31: y-dimension
    bits 32-end: image data
    rest of bits are padded zero
    """
    return out[:int(4e5)]  # cut off any extra values
