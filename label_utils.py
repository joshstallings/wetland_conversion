"""
One place to binarize the raw three valued NLCD label, used by the population
scan, and both dataset streams. 
"""

import numpy as np


def binarize_label(raw_label):
    """raw_label: array-like of {0, 1, 2}. Returns float32 array of {0.0, 1.0}."""
    return (np.asarray(raw_label) == 1).astype(np.float32)
