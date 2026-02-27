import numpy as np

BITS = 2

# maps sets of 2 bits to a 4-qam symbol with d = 1 by default
def bit_to_symbol(bits: np.ndarray, d: float = 1) -> np.ndarray:
    assert len(bits) % BITS == 0, "non-multiple of 4 bits length"
    
    num_symbols = len(bits) // BITS
    
    out = np.zeros(num_symbols, dtype = 'complex')
    
    for i in range(num_symbols):
        curr = bits[i * BITS : BITS * i + BITS]
        print(curr)
        
        if (np.array_equal(curr[0], 0)):
            out[i] -= 0.5 * d
        else:
            out[i] += 0.5 * d
            
        if (np.array_equal(curr[1], 0)):
            out[i] -= 0.5j * d
        else:
            out[i] += 0.5j * d

    return out

print(bit_to_symbol([1,1,1,0,0,1,0,0]))

"""
16 QAM logic
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
"""