import numpy as np

def bit_to_symbol(bits: np.ndarray, d: float = 1) -> np.ndarray:
    assert len(bits) % 4 == 0, "non-multiple of 4 bits length"
    
    num_symbols = len(bits) // 4
    
    out = np.empty(num_symbols, dtype = 'complex')
    
    for i in range(num_symbols):
        curr = bits[i * 4 : 4 * (i + 1)]
        if np.array_equal(curr[0:2], [0,0]):
            out[i] = -1.5 * d
        elif np.array_equal(curr[0:2], [0,1]):
            out[i] = -.5 * d
        elif np.array_equal(curr[0:2], [1,1]):
            out[i] = .5 * d
        elif np.array_equal(curr[0:2], [1,0]):
            out[i] = 1.5 * d
        
        if np.array_equal(curr[2:4], [0,0]):
            out[i] += -1.5j * d
        elif np.array_equal(curr[2:4], [0,1]):
            out[i] += -.5j * d
        elif np.array_equal(curr[2:4], [1,1]):
            out[i] += .5j * d
        elif np.array_equal(curr[2:4], [1,0]):
            out[i] += 1.5j * d
            
    return out
