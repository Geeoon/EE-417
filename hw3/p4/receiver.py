# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

from symbol_detector_to_bits import symbol_detector_to_bits
from bit_to_symbol import bit_to_symbol

def bits_to_val(bits: np.ndarray) -> int:
    """
    MSB is index 0

    :param bits: the bits to convert
    :type bits: np.ndarray
    :return: the value converted from bits
    :rtype: int
    """
    out = 0
    for bit in bits:
        out <<= 1
        out |= int(bit)
    return out

def receiver(recvd: np.ndarray, preamble: np.ndarray, bits_per_symbol: int=1, amplitude: float=2, symbol_size: int=1) -> np.ndarray:
    """
    converts a received signal to the raw data
    
    :param input: the received signal
    :type input: np.ndarray
    :param preamble: the preamble to search for
    :type preamble: np.ndarray
    :param bits_per_symbol: the number of bits encoded in each symbol
    :type input: int
    :param amplitude: the amplitude of the received signal
    :type amplitude: float

    :return: the reconstructed image
    :rtype: ndarray
    """
    assert(bits_per_symbol >= 1)
    preamble_symbols = bit_to_symbol(np.repeat(preamble, symbol_size))
    # find preamble index
    # convolve signal
    convolved = np.correlate(recvd, preamble_symbols, mode='valid')
    # find index of max
    # first occurance above 97.5% match with preamble, based on correlation
    correlation = np.abs(np.sum(preamble_symbols ** 2))
    indices = np.where(np.abs(convolved) > correlation * .975)[0]
    
    if len(indices) == 0:
        index = np.argmax(convolved)
    else:
        index = indices[0]

    # convert symbol to value
    out = symbol_detector_to_bits(recvd)
    
    # parse x and y
    out = out[index*4+len(preamble):]
    x = bits_to_val(out[:16])
    y = bits_to_val(out[16:32])
    print("x:", x, out[:16])
    print("y:", y, out[16:32])
    out = out[32:]

    # return reconstructed image
    if (x < 1 or y < 1) or (len(out) < x * y):
        return None, index
    out = out[:x*y].astype(np.uint8)
    out = out.reshape((x, y))  
    
    return out, index
