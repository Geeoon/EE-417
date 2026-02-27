import numpy as np
from bit_to_symbol import bit_to_symbol
from transmitter import to_symbol

BITS = 2

def symbol_detector_to_bits(symbols: np.ndarray, d: float = 1) -> np.ndarray:
    num_bits = len(symbols) * BITS
    
    out = np.zeros(num_bits)

    diffs = []
    for i in range(pow(BITS, 2)):
        sym_arr = np.tile(bit_to_symbol(to_symbol(i, out_len=BITS)), len(symbols))
        diffs.append(np.abs(sym_arr - symbols))
    
    # find minimum for each index
    arg_mins = np.argmin(diffs, axis=0)
    for i, min in enumerate(arg_mins):
        out[i*BITS : i*BITS+BITS] = to_symbol(min, out_len=BITS) // 1
    return out

print(symbol_detector_to_bits([ 0.5+0.5j,  0.5-0.5j, -0.5+0.5j, -0.5-0.5j]))