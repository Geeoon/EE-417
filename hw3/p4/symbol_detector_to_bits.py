import numpy as np

def symbol_detector_to_bits(symbols: np.ndarray, d: float = 1) -> np.ndarray:
    num_bits = len(symbols) * 4
    
    bits = np.empty(num_bits)
    
    for i in range(len(symbols)):
        curr = symbols[i]
        curr_real = np.real(curr)
        curr_imag = np.imag(curr)
        
        if curr_real > 0:
            dist_11 = abs(curr_real - 0.5 * d)
            dist_10 = abs(curr_real - 1.5 * d)
            
            bits[i * 4: i * 4 + 2] = [1, 1] if (dist_11 <= dist_10) else [1, 0]
        else:
            dist_00 = abs(curr_real + 0.5 * d)
            dist_01 = abs(curr_real + 1.5 * d)
            
            bits[i * 4: i * 4 + 2] = [0, 0] if (dist_00 <= dist_01) else [0, 1]
                
        if curr_imag > 0:
            dist_11 = abs(curr_imag - 0.5 * d)
            dist_10 = abs(curr_imag - 1.5 * d)
            
            bits[i * 4 + 2: i * 4 + 4] = [1, 1] if (dist_11 <= dist_10) else [1, 0]
        else:
            dist_00 = abs(curr_imag + 1.5 * d)
            dist_01 = abs(curr_imag + 0.5 * d)
            
            bits[i * 4 + 2: i * 4 + 4] = [0, 0] if (dist_00 <= dist_01) else [0, 1]
    
    return bits