import numpy as np

def _bits_to_val(input: list[int]) -> int:
    """
    :param input: MSB is at the start
    """
    out = 0
    for index, bit in enumerate(reversed(input)):
        out += bit * 2**index
    return out


def convolution_encoder(input: np.ndarray, G: list[list[int]]=[[0o5, 0o7]], pad_ending=True) -> np.ndarray:
    """
    Convolution encoder for Trellis
    :param input: the input bits
    :param G: the G matrix to use
    :return: the coded bits
    """
    assert input.dtype == np.uint8, "Input list must be uin8 type"
    assert np.max(input) == 1 and np.min(input) == 0, "Input list must only contain 1s and 0s"
    if pad_ending:
        # find the number of states
        pads = -1
        for val in G:
            for g in val:
                if pads < g.bit_length():
                    pads = g.bit_length()
        input = np.pad(input, (0, pads), 'constant')

    outputs = []  # to be xored together
    for input_idx, val in enumerate(G):
        outputs.append([])
        for t in range(1, len(input) + 1):
            for output_idx, g in enumerate(val):
                depth = g.bit_length()
                outputs[-1].append((_bits_to_val(input[max(t-depth,0):t]) & g).bit_count() % 2)

    out = []
    for i in range(len(outputs[0])):
        out.append(0)
        for output in outputs:
            out[-1] = out[-1] + output[i]
        out[-1] %= 2
    return out

print(convolution_encoder(input=np.array([0, 1, 0, 1, 1, 0, 1], dtype=np.uint8)))
