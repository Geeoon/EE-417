import numpy as np

# uses hard decoding to output the original bit stream
def hard_decoder_regular(bits: np.ndarray, d = (float) 1) -> np.ndarray:
    path = []
    
    state = 0b00
    
    for i in range(len(bits), 2):
        curr = bits[i: i + 2]
        
        if state == 0:
            if np.array_equal(curr, [0, 0]):
                path.append(0)
            elif np.array_equal(curr, [1, 1]):
                path.append(1)
        elif state == 1:
            if np.array_equal(curr, [0, 0]):
                path.append(0)
            elif np.array_equal(curr, [1, 1]):
                path.append(1)
        elif state == 2:
            if np.array_equal(curr, [0, 0]):
                path.append(0)
            elif np.array_equal(curr, [1, 1]):
                path.append(1)
        elif state == 3:
            if np.array_equal(curr, [0, 0]):
                path.append(0)
            elif np.array_equal(curr, [1, 1]):
                path.append(1)
        
        state = (state << 1 | bit) % 4