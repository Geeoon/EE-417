import numpy as np

from bit_to_symbol_mapper import bit_to_symbol_mapper, symbol_to_bit_mapper
from convolutional_encoder import convolution_encoder

def _bits_to_val(input: list[int]) -> int:
    """
    :param input: MSB is at the start
    """
    out = 0
    for index, bit in enumerate(reversed(input)):
        out += bit * int(2**index)
    return int(out)

def _expected_output(state: int, input: int, G: list[list[int]]=[[0o5, 0o7]]) -> np.ndarray:
    out = np.zeros_like(G[0], dtype=np.uint8)
    # print(state, input)
    for i, g in enumerate(G):  # for each input bit
        for j, val in enumerate(g):  # for each output bit
            out[j] = ((((state << 1) | ((input & (1 << i)) >> i))) & val).bit_count() & 1
    return out

def convolutional_hard_decoder(symbols: np.ndarray, G: list[list[int]]=[[0o5, 0o7]]) -> np.ndarray:
    num_inputs = len(G)
    output_size = len(G[0])
    num_states = -1
    num_bits = -1
    for g in G:
        for val in g:
            num_bits = max(num_bits, val.bit_length() - 1)
    num_states = 2 ** num_bits
    assert num_bits > 0, "G matrix malformed"

    # precompute expected outputs and next state
    expected_output = np.zeros((num_states, 2**num_inputs), dtype=np.uint8)
    next_states = np.zeros((num_states, 2**num_inputs), dtype=np.uint8)
    for state in range(num_states):
        for inp in range(2**num_inputs):
            bits = _bits_to_val(_expected_output(state, inp, G=G))
            expected_output[state, inp] = bits
            next_states[state, inp] = ((state << num_inputs) | inp) & (num_states - 1)

    # convert to bits
    bits = symbol_to_bit_mapper(symbols)
    assert (len(bits) % num_inputs) == 0, "Number of received bits is not a multiple of the number of inputs"

    # initial state
    transitions = [[{ "state_weight": 0.0, "previous": None }]]
    for _ in range(1, num_states):
        transitions[0].append({ "state_weight": float('inf'), "previous": None })

    # for each input, calculate weights of previous states' transitions then find the weight of the current states
    for i in range(len(symbols)):  # for each of the new inputs
        # print(f"t = {i+1}")
        inputs = np.array(bits[i*output_size:(i+1)*output_size])
        # print(inputs, _bits_to_val(inputs))
        for j in range(len(transitions[-1])):  # for each previous state (index)
            # print(f"  for previous state {j}")
            for k in range(2 ** num_inputs):  # for each possible transition
                # NOTE: this part is specific to hard decoding
                transitions[-1][j][k] = { "weight": (_bits_to_val(inputs) ^ expected_output[j, k]).bit_count(), "next_state": next_states[j, k] }  # weight, number of 1 bits
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
                    if lowest['weight'] >= contender_weight:
                        # found lower weight
                        lowest['state'] = old_state
                        lowest['weight'] = contender_weight
                        lowest['transition'] = transition
            # assert lowest['state'] is not None, "Could not find an old state that points to the new state"
            transitions[-1].append({ "state_weight": lowest['weight'], "previous": { "state": lowest['state'], "transition": lowest['transition'] } })

    # print for debugging
    # for i, transition in enumerate(transitions):
    #     print(f"t = {i}:")
    #     for j, state in enumerate(transition):
    #         print(f"    state = {j}: {state}")

    # traverse backwards from last t to find minimum weight path
    out = [transitions[-1][0]['previous']['transition']]  # the input before that got us to the final 0 state
    prev_state = transitions[-1][0]['previous']['state']  # the state before that got us to the final 0 state
    for transition in transitions[1:-1][::-1]:
        out.insert(0, transition[prev_state]['previous']['transition'])
        prev_state = transition[prev_state]['previous']['state']

    return out
