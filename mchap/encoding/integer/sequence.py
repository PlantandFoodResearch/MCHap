#!/usr/bin/env python3

import numpy as np

__all__ = [
    "is_gap",
    "is_call",
    "is_valid",
    "argsort",
    "sort",
    "depth",
]


def is_gap(array):
    """Identify gap positions in an integer encoded biological sequence.

    Parameters
    ----------
    array : ndarray, int
        Array of integers encoding alleles.

    Returns
    -------
    mask : ndarray, bool
        Array of booleans indicating gap positions.

    Notes
    -----
    Gaps are encoded as the value `-1`.

    """
    return array == -1


def is_call(array):
    """Identify non-gap positions in an integer encoded biological sequence.

    Parameters
    ----------
    array : ndarray, int
        Array of integers encoding alleles.

    Returns
    -------
    mask : ndarray, bool
        Array of booleans indicating non-gap positions.

    Notes
    -----
    Gaps are encoded as the value `-1`.

    """
    return array >= 0


def is_valid(array):
    """Identify positions of valid values in an integer encoded biological sequence.

    Parameters
    ----------
    array : ndarray, int
        Array of integers encoding alleles.

    Returns
    -------
    mask : ndarray, bool
        Array of booleans indicating valid positions.

    Notes
    -----
    Alleles are encoded as values >- `0` and gaps are encoded as the value `-1`.

    """
    return array >= -1


def argsort(array):
    """Argsort a set of biological sequences that are encoded as integers.

    Parameters
    ----------
    array : ndarray, int, shape (n_sequences, n_positions)
        2D array of integers encoding a series of biological sequences.

    Returns
    -------
    index_array : ndarray, int, shape (n_sequences, )
        Array of integer indices indicating sorted order of sequences.

    """
    assert array.ndim == 2
    return np.lexsort(np.flip(array, axis=-1).transpose((-1, -2)))


def sort(array):
    """Sort a set of biological sequences that are encoded as integers.

    Parameters
    ----------
    array : ndarray, int, shape (n_sequences, n_positions)
        2D array of integers encoding a series of biological sequences.

    Returns
    -------
    array : ndarray, int, shape (n_sequences, n_positions)
        2D sorted array of integers encoding a series of biological sequences.

    """
    return array[argsort(array)]


def depth(array, counts=None):
    """Position-wise depth of a set of biological sequences that are encoded as integers.

    Parameters
    ----------
    array : ndarray, int, shape (n_sequences, n_positions)
        2D array of integers encoding a series of biological sequences.

    Returns
    -------
    depth : ndarray, int, shape (n_positions, )
        1D array of integer depth per position.

    Notes
    -----
    Gap values (`-1`) do not count towards depth.

    """
    if counts is None:
        return np.sum(is_call(array), axis=-2)
    else:
        counts = np.expand_dims(counts, -1)
        return np.sum(is_call(array).astype(int) * counts, axis=-2)
