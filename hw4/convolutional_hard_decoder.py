import numpy as np

from bit_to_symbol_mapper import bit_to_symbol_mapper, symbol_to_bit_mapper
from convolutional_encoder import convolution_encoder

def bits_to_val(bits: np.ndarray) -> int:
    """
    MSB is index 0

    :param bits: the bits to convert
    :type bits: np.ndarray
    :return: the value converted from bits
    :rtype: int
    """
    out = 0
    for bit in bits:
        out <<= 1
        out |= int(bit)
    return out


def convolutional_hard_decoder(symbols: np.ndarray, G: list[list[int]]=[[0o5, 0o7]]) -> np.ndarray:
    num_inputs = len(G)
    num_states = -1
    num_bits = -1
    for g in G:
        for val in g:
            num_bits = max(num_bits, val.bit_count() - num_inputs)
    num_states = 2 ** num_bits

    # convert to bits
    bits = symbol_to_bit_mapper(symbols)
    assert (len(bits) % num_inputs) == 0, "Number of received bits is not a multiple of the number of inputs"

    # initial state
    transitions = [[{ "state_weight": 0.0, "previous": None }]]
    for _ in range(1, num_states):
        transitions[0].append({ "state_weight": float('inf'), "previous": None })

    # for each input, calculate weights of previous states' transitions then find the weight of the current states
    for i in range(len(symbols)):  # for each of the new inputs
        inputs = np.array(bits[i*2:i*2+num_inputs])
        for j in range(len(transitions[-1])):  # for each previous state (index)
            for k in range(2 ** num_inputs):  # for each possible transition
                # NOTE: this next line part is specific to hard decoding
                transitions[-1][j][k] = { "weight": (bits_to_val(inputs) ^ k).bit_count(), "next_state": ((j << num_inputs) | k) & (num_states - 1) }  # weight, number of 1 bits
        transitions.append([])
        for new_state in range(num_states):  # for each new state
            # find state with lowest state weight + transition weight to this new state
            lowest = { 'state': None, 'weight': float('inf'), 'transition': 0 }
            for old_state in range(len(transitions[-2])):  # for each old state
                for transition in range(2 ** num_inputs):  # for each old state's transition
                    # check if the state points to the new state
                    if transitions[-2][old_state][transition]['next_state'] != new_state:
                        continue
                    # compute possible weight for the new state
                    contender_weight = transitions[-2][old_state]['state_weight'] + transitions[-2][old_state][transition]['weight']
                    if lowest['weight'] > contender_weight:
                        # found lower weight
                        lowest['state'] = old_state
                        lowest['weight'] = contender_weight
                        lowest['transition'] = transition
            # assert lowest['state'] is not None, "Could not find an old state that points to the new state"
            transitions[-1].append({ "state_weight": lowest['weight'], "previous": { "state": lowest['state'], "transition": lowest['transition'] } })
    

    for i, transition in enumerate(transitions):
        print(f"t = {i}:")
        for j, state in enumerate(transition):
            print(f"    state = {j}: {state}")

    # traverse backwards from last t to find minimum weight path
    out = [transitions[-1][0]['previous']['transition']]  # the input before that got us to the final 0 state
    prev_state = transitions[-1][0]['previous']['state']  # the state before that got us to the final 0 state
    for transition in transitions[1:-1][::-1]:
        out.insert(0, transition[prev_state]['previous']['transition'])
        prev_state = transition[prev_state]['previous']['state']

    return out

test_encoding = convolution_encoder(np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8))
test_symbols = bit_to_symbol_mapper(test_encoding)
# print(test_encoding, len(test_encoding), test_symbols)
print(convolutional_hard_decoder(test_symbols))
