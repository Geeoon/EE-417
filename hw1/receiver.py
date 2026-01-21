# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

def receiver(recvd: np.ndarray, bits_per_symbol: int=1, amplitude: float=2) -> np.ndarray:
    """
    converts a received signal to the
    
    :param input: the received signal
    :type input: np.ndarray
    :param bits_per_symbol: the number of bits encoded in each symbol
    :type input: int
    :param amplitude: the amplitude of the received signal
    :type amplitude: float

    :return: Description
    :rtype: ndarray
    """
    assert(bits_per_symbol >= 1)

    out = np.rint(((recvd + amplitude / 2) / amplitude) * ((2 ** bits_per_symbol) - 1))
    assert(len(out) == len(recvd))
    return out.astype(np.int64)
