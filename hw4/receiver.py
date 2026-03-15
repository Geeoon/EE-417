# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

from convolutional_encoder import convolution_encoder
from convolutional_hard_decoder import convolutional_hard_decoder
from convolutional_soft_decoder import convolutional_soft_decoder

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
    preamble_symbols = convolution_encoder(preamble)

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
    
    # hard decoder
    out_hard = convolutional_hard_decoder(recvd[index:])
    out_soft = convolutional_soft_decoder(recvd[index:])

    # parse x and y hard
    x = bits_to_val(out_hard[:16])
    y = bits_to_val(out_hard[16:32])
    out_hard = out_hard[32:]
    
    # return reconstructed image
    if not ((x < 1 or y < 1) or (len(out_hard) < x * y)):
        out_hard = out_hard[:x*y].astype(np.uint8)
        out_hard = out_hard.reshape((x, y))
    
    # parse x and y soft
    x = bits_to_val(out_soft[:16])
    y = bits_to_val(out_soft[16:32])
    out_soft = out_soft[32:]
    
    if not ((x < 1 or y < 1) or (len(out_soft) < x * y)):
        out_soft = out_soft[:x*y].astype(np.uint8)
        out_soft = out_soft.reshape((x, y))
    
    return out_hard, out_soft, index
