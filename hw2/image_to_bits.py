# Geeoon Chung and Atharv Dixit
# EE 417 Winter 2026

import cv2
import numpy as np

def image_to_bits(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, 0)  # read as black and white
    return image >> 7  # quantize to 1 bit
