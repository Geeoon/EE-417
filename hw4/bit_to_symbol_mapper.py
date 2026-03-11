import numpy as np

def bit_to_symbol_mapper(bits: np.ndarray, d: float = 1) -> np.ndarray:
    assert len(bits) % 2 == 0, "bits must be a multiple of 2 bits"
    assert bits.dtype == np.uint8, "bits must be of type uint8"
    assert np.max(bits) <= 1 and np.min(bits) >= 0, "bits must be either 1 or 0"
    out = np.zeros(len(bits) // 2, dtype=np.complex128)
    out -= .5 +.5j
    out += bits[::2]
    out += bits[1::2] * 1j
    return out

print(bit_to_symbol_mapper(np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)))
