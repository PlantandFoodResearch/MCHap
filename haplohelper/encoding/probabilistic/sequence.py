#!/usr/bin/env python3

import numpy as np
import numba


@numba.njit
def _is_gap_vec(vector):
    return np.all(np.isnan(vector))


@numba.njit
def _is_base_vec(vector, precision):
    # must be a sequence of numbers suming to 1
    # may be followed by 0 or more nans
    total = 0.0
    n = len(vector)

    # first add numbers
    for i in range(n):
        if np.isnan(vector[i]):
            break
        else:
            total += vector[i]

    total = np.round(total, precision)
    if total != 1.0:
        return False

    # now check any remaining values are all nan
    for j in range(i + 1, n):
        if np.isnan(vector[j]):
            pass
        else:
            return False
    
    return True


@numba.njit
def _is_valid_vec(vector, precision):
    return _is_base_vec(vector, precision) or _is_gap_vec(vector)


@numba.jit
def _is_gap(array, result):
    for i in range(len(result)):
        result[i] = _is_gap_vec(array[i])


@numba.jit
def _is_base(array, result, precision):
    for i in range(len(result)):
        result[i] = _is_base_vec(array[i], precision)


@numba.jit
def _is_valid(array, result, precision):
    for i in range(len(result)):
        result[i] = _is_valid_vec(array[i], precision)


def is_gap(array):
    shape = array.shape[0: -1]
    if shape is ():
        return _is_gap_vec(array)
    length = np.prod(shape)
    result = np.empty(length, np.bool)
    _is_gap(array.reshape(length, -1), result)
    return result.reshape(shape)


def is_base(array, precision=None):
    if precision is None:
        precision = np.finfo(np.float).precision
    shape = array.shape[0: -1]
    if shape is ():
        return _is_base_vec(array, precision)
    length = np.prod(shape)
    result = np.empty(length, np.bool)
    _is_base(array.reshape(length, -1), result, precision)
    return result.reshape(shape)


def is_valid(array, precision=None):
    if precision is None:
        precision = np.finfo(np.float).precision - 2
    shape = array.shape[0: -1]
    if shape is ():
        return _is_valid_vec(array, precision)
    length = np.prod(shape)
    result = np.empty(length, np.bool)
    _is_valid(array.reshape(length, -1), result, precision)
    return result.reshape(shape)


def is_known(array, p=0.95):
    return np.any(array >= p, axis=-1)


def depth(array, counts=None, precision=None):
    """Calculate sequence depth of one or more biological 
    sequences that are encoded as probabilistic row vectors.
    Gaps (nan vectors) don't count towards sequence depth.
    """
    if array.ndim == 2:
        array = np.expand_dims(array, 0)

    array = is_base(array, precision=precision).astype(np.int)
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
