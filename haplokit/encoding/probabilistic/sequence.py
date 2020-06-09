#!/usr/bin/env python3

import numpy as np
import numba


def is_gap(array):
    """Identify gap positions in a biological sequence 
    encoded as probabilistic vectors.

    Parameters
    ----------
    array : ndarray, float, shape(n_positions, max_allele)
        Array of vectors encoding allele probabilities

    Returns
    -------
    mask : ndarray, bool, shape(n_positions, )
        Array of booleans indicating gap positions

    Notes
    -----
    Gap vectors contain `nan` values and are possibly 
    padded with `0` values.

    """
    # gaps may still be zero padded
    bools = np.logical_or(np.isnan(array), array == 0)
    return np.all(bools, axis=-1)


def is_dist(array, precision=None):
    """Identify valid allele distributions in a biological sequence 
    encoded as probabilistic vectors.

    Parameters
    ----------
    array : ndarray, float, shape(n_positions, max_allele)
        Array of vectors encoding allele probabilities.
    precision : int, optional
        Decimal precision to use when checking sum of probabilities.

    Returns
    -------
    mask : ndarray, bool, shape(n_positions, )
        Array of booleans indicating distribution positions.

    Notes
    -----
    Valid distributions contain values that sum to `1` and
    do not contatin `nan` values.

    """
    if precision is None:
        precision = np.finfo(np.float).precision
    return np.round(np.sum(array, axis=-1), precision) == 1


def is_valid(array, precision=None):
    """Identify valid allele vectors in a biological sequence 
    encoded as probabilistic vectors.
    This includes distributions and gaps.

    Parameters
    ----------
    array : ndarray, float, shape(n_positions, max_allele)
        Array of row vectors encoding allele probabilities.
    precision : int, optional
        Decimal precision to use when checking sum of probabilities.

    Returns
    -------
    mask : ndarray, bool, shape(n_positions, )
        Array of booleans indicating positions of valid vectors.

    Notes
    -----
    Invlaid vectors include probability vectors that do not sum to `1`
    and vectors that include a mixture of `nan` values and none-`0`
    probabilities.

    """
    return np.logical_or(
        is_dist(array, precision=precision), 
        is_gap(array)
    )


def depth(array, counts=None, precision=None):
    """Position-wise depth of a set of biological sequences 
    encoded as probabilistic vectors.

    Parameters
    ----------
    array : ndarray, float, shape (n_sequences, n_positions, max_allele)
        3D array of row vectors encoding allele probabilities.
    precision : int, optional
        Decimal precision to use when checking sum of probabilities.

    Returns
    -------
    depth : ndarray, float, shape (n_positions, )
        1D array of depth per position.

    Notes
    -----
    Gap values do not count towards depth.

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
    """Position-wise allele depth of a set of biological sequences 
    encoded as probabilistic vectors.

    Parameters
    ----------
    array : ndarray, float, shape (n_sequences, n_positions, max_allele)
        3D array of row vectors encoding allele probabilities.
    counts : ndarray, int, shape (n_sequences, ), optional
        array of counts of each sequence.

    Returns
    -------
    allele_depth : ndarray, float, shape (n_positions, max_allele)
        2D array of expected depth per position per allele.

    Notes
    -----
    Gap values do not count towards depth.

    """
    if array.ndim == 2:
        array = np.expand_dims(array, 0)

    if counts is None:
        return np.nansum(array, axis=-3)
    else:
        return np.nansum(array * counts.reshape(-1, 1, 1), axis=-3)
