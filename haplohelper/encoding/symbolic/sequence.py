import numpy as np


def is_gap(array, gap='-'):
    return array == gap


def depth(array, gap='-'):
    return np.sum(array != gap, axis=0)

