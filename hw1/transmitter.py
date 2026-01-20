import numpy as np

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
    
    out = (input_signal * (amplitude / ((2 ** bits_per_symbol) - 1))) - amplitude / 2
    out = np.repeat(out, symbol_size)
    return out
