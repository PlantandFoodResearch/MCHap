#!/usr/bin/env python3

import numpy as np


def is_gap(array):
    return array == -1


def is_base(array):
    return array >= 0


def is_valid(array):
    return array >= -1


def argsort(array):
    """Argsort a set of biological sequences that are encoded as scalars.
    """
    assert array.ndim == 2
    return np.lexsort(np.flip(array, axis=-1).transpose((-1, -2)))


def sort(array):
    """Sort a set of biological sequences that are encoded as scalars.
    """
    return array[argsort(array)]
