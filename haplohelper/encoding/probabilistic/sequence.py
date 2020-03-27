#!/usr/bin/env python3

import numpy as np
import numba


def is_gap(array):
    """Gaps are not expressed without nans"""
    return np.all(np.isnan(array), axis=-1)


def is_dist(array, precision=None, nan_padded=False):
    if precision is None:
        precision = np.finfo(np.float).precision
    if nan_padded:
        return np.round(np.nansum(array, axis=-1), precision) == 1
    else:
        return np.round(np.sum(array, axis=-1), precision) == 1


def is_valid(array, precision=None, nan_padded=False):
    return np.logical_or(
        is_dist(array, precision, nan_padded), 
        is_gap(array)
    )


def depth(array, counts=None, precision=None):
    """Calculate sequence depth of one or more biological 
    sequences that are encoded as probabilistic row vectors.
    Gaps (nan vectors) don't count towards sequence depth.
    """
    if array.ndim == 2:
        array = np.expand_dims(array, 0)

    array = is_dist(array, precision=precision).astype(np.int)
    if counts:
        array *= counts

    if array.ndim == 1:
        return array
    else:
        return np.sum(array, axis=0)


def allele_depth(array, counts=None):
    """Calculate allele depth of one or more biological 
    sequences that are encoded as probabilistic row vectors.
    Gaps (nan vectors) don't count towards allele depth.
    """
    if array.ndim == 2:
        array = np.expand_dims(array, 0)

    if counts is None:
        return np.nansum(array, axis=-3)
    else:
        return np.nansum(array * counts.reshape(-1, 1, 1), axis=-3)
