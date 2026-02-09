# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

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

def receiver(recvd: np.ndarray, preamble: np.ndarray, bits_per_symbol: int=1, amplitude: float=2) -> np.ndarray:
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

    preamble = (preamble * (amplitude / ((2 ** bits_per_symbol) - 1))) - amplitude / 2

    # find preamble index
    # convolve signal
    convolved = np.convolve(recvd, preamble, mode='valid')
    # find index of max
    index = np.argmax(convolved)

    # convert symbol to value
    out = np.rint(((recvd + amplitude / 2) / amplitude) * ((2 ** bits_per_symbol) - 1))
    
    # parse x and y
    out = out[index+len(preamble)+3:]
    x = bits_to_val(out[:16])
    y = bits_to_val(out[16:32])
    out = out[32:]

    # return reconstructed image
    assert len(out) >= x * y
    out = out[:x*y].astype(np.uint8)
    out = out.reshape((x, y))
    return out
