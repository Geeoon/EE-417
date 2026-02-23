import numpy as np
from bit_to_symbol import bit_to_symbol
from transmitter import to_symbol

def symbol_detector_to_bits(symbols: np.ndarray, d: float = 1) -> np.ndarray:
    num_bits = len(symbols) * 4
    
    out = np.empty(num_bits)

    diffs = []
    for i in range(16):
        sym_arr = np.tile(bit_to_symbol(to_symbol(i, out_len=4)), len(symbols))
        diffs.append(np.abs(sym_arr - symbols))
    
    # find minimum for each index
    arg_mins = np.argmin(diffs, axis=0)
    for i, min in enumerate(arg_mins):
        out[i*4 : i*4+4] = to_symbol(min, out_len=4) // 1
    return out
