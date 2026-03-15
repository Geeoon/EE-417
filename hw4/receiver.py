# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import numpy as np

from convolutional_encoder import convolution_encoder
from convolutional_hard_decoder import convolutional_hard_decoder
from convolutional_soft_decoder import convolutional_soft_decoder

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

def receiver(recvd: np.ndarray, preamble: np.ndarray, expected_preamble_idx: int) -> np.ndarray:
    """
    converts a received signal to the raw data
    
    :param input: the received signal
    :param preamble: the preamble to search for
    :param expected_preamble_idx: the expected preamble index
    

    :return: the reconstructed image
    """
    preamble_symbols = convolution_encoder(preamble, pad_ending=False)
    # find preamble index
    # convolve signal
    convolved = np.correlate(recvd, preamble_symbols, mode='valid')
    
    # find index of max
    # first occurance above 97.5% match with preamble, based on correlation
    preamble_symbols_energy = np.sum(np.abs(preamble_symbols) ** 2)
    threshold = preamble_symbols_energy * 0.975
    indices = np.where(np.abs(convolved) > threshold)[0]
    
    if len(indices) == 0:
        index = np.argmax(np.abs(convolved))
    else:
        index = indices[0]

    print("preamble detected at: ", index)
    if index != expected_preamble_idx:
        return None, None, index

    # hard decoder
    out_hard = np.array(convolutional_hard_decoder(recvd[index:index+48]))
    print(out_hard)
    # soft decoder
    out_soft = np.array(convolutional_soft_decoder(recvd[index:index+48]))

    # check length of received values vs decoded values
    print("post-preamble received length: ", len(recvd[index:]))
    print("post-preamble hard-decoded length: ", len(out_hard))
    print("post-preamble soft-decoded length: ", len(out_soft))

    # parse x and y hard
    x_hard = bits_to_val(out_hard[len(preamble):len(preamble)+16])
    y_hard = bits_to_val(out_hard[len(preamble)+16:len(preamble)+32])
    out_hard = out_hard[len(preamble)+32:]
    
    # parse x and y soft
    x_soft = bits_to_val(out_soft[len(preamble):len(preamble)+16])
    y_soft = bits_to_val(out_soft[len(preamble)+16:len(preamble)+32])
    out_soft = out_soft[len(preamble)+32:]

    print("hard-decoded (x, y): ", x_hard, y_hard)
    print("soft-decoded (x, y): ", x_soft, y_soft)

    # now do the entire thing
    out_hard = np.array(convolutional_hard_decoder(recvd[index:index + len(preamble) + 48 + x_hard * y_hard + 2]))
    out_soft = np.array(convolutional_soft_decoder(recvd[index:index + len(preamble) + 48 + x_soft * y_soft + 2]))
    out_hard = out_hard[len(preamble) + 48:]
    out_soft = out_soft[len(preamble) + 48:]


    # reshape the hard-decoded image to the transmitted dimensions
    if not (x_hard < 1 or y_hard < 1) or not (len(out_hard) < x_hard * y_hard):
        out_hard = out_hard[:x_hard * y_hard].astype(np.uint8)
        out_hard = np.reshape(out_hard, (x_hard, y_hard))

    # reshape the soft-decoded image to the transmitted dimensions
    if not (x_soft < 1 or y_soft < 1) or not (len(out_soft) < x_soft * y_soft):
        out_soft = out_soft[:x_soft * y_soft].astype(np.uint8)
        out_soft = np.reshape(out_soft, (x_soft, y_soft))

    return out_hard, out_soft, index
