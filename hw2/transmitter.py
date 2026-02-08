# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

def to_symbol(val: int, bits_per_symbol: int=1, out_len: int=16) -> np.ndarray:
    assert val < 2 ** (out_len * bits_per_symbol), "output length isn't large enough to hold all bits"
    assert val >= 0, "can't convert negative numbers"
    out = np.array([])
    while val:
        out = np.append(out, val % (2 ** bits_per_symbol))
        val >>= bits_per_symbol
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
    assert symbol_size >= 1
    assert bits_per_symbol >= 1
    assert amplitude >= 1
    assert np.max(input_signal) == 1 or np.max(input_signal) == 0  # make sure it's a 1 bit stream
    # add length and x, y dimensions to front of message
    input_signal = np.concatenate([to_symbol(len(input_signal), bits_per_symbol=bits_per_symbol),
                                   to_symbol(input_signal.shape[0], bits_per_symbol=bits_per_symbol),
                                   to_symbol(input_signal.shape[1], bits_per_symbol=bits_per_symbol),
                                   input_signal.flatten()])
    input_signal = np.concatenate([(1, 0, 1, 0, 1, 1, 1, 1), input_signal])  # add preamble
    # pad with zeros
    input_signal = np.concatenate((input_signal, (0,) * int(1e5 - len(input_signal))))
    out = (input_signal * (amplitude / ((2 ** bits_per_symbol) - 1))) - amplitude / 2
    out = np.repeat(out, symbol_size)
    
    # make sure the output is the right size
    assert len(out) >= 1e5, f"{len(out)} < 1e5"
    
    """
    So the output looks as such
    bits 0-7: preamble
    bits 8-39: total length in symbols
    bits 40-71: x-dimension
    bits 72-103: y-dimension
    bits 104-end: image data
    rest of bits are padded zero
    """
    return out[:int(1e5)]  # cut off any extra values

# from image_to_bits import image_to_bits
# im = image_to_bits('./photos/monalisa_diff.png')
# out = transmitter(im, bits_per_symbol=1, symbol_size=1)
# print(out)
