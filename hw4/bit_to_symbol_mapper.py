import numpy as np

def bit_to_symbol_mapper(bits: np.ndarray, d: float = 1) -> np.ndarray:
    assert len(bits) % 2 == 0, "bits must be a multiple of 2 bits"
    assert bits.dtype == np.uint8, "bits must be of type uint8"
    assert np.max(bits) <= 1 and np.min(bits) >= 0, "bits must be either 1 or 0"
    out = np.zeros(len(bits) // 2, dtype=np.complex128)
    out -= (d/2) + ((d/2) * 1j)
    out += bits[::2] * d
    out += bits[1::2] * (d*1j)
    return out

def symbol_to_bit_mapper(symbols: np.ndarray, d: float = 1) -> np.ndarray:
    assert symbols.dtype == np.complex128, "symbols must be complex"
    i_bits = (symbols.real + (d/2)).astype(int)
    q_bits = (symbols.imag + (d/2)).astype(int)
    return np.ravel(np.column_stack((i_bits, q_bits)))
