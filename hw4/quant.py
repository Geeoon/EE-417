# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import matplotlib.pyplot as plt
import numpy as np

from scipy import special as sp

def q_function(snr):
    value = np.sqrt(pow(10, snr / 10) / 5)
    return 0.5 - 0.5*sp.erf(value/np.sqrt(2))


