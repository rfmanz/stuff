"""
Modelling util functions.
"""

# Authors: Guillermo Navas-Palencia <guillermo.navas@bbva.com>
# BBVA - Copyright 2018.

import numpy as np


def check_consistent_length(*arrays):
    """
    https://github.com/scikit-learn/scikit-learn/blob/bac89c253b35a8f1a3827389fbee0f5bebcbc985/sklearn/utils/validation.py
    """
    array_samples = [X.shape[0] for X in arrays if X is not None]
    if len(np.unique(array_samples)) > 1:
        raise ValueError("Found input variables with inconsistent number of "
                         "samples: {}".format([int(s) for s in array_samples]))
