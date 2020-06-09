#!/usr/bin/env python3

import numpy as np
import numba

from haplokit.assemble.util import random_choice as _random_choice


def call_alleles(array, p=0.95, dtype=np.int8):
    """Call allele at each position if it's probability is
    greater than or equal to the specified value.

    Parameters
    ----------
    array : ndarray, float
        Array of row vectors encoding allele probabilities.
    p : float, optional
        Minimum probability required to make a call.
    dtype : dtype, optional
        Specify dtype of returned array.

    Returns
    -------
    array : int
        Array of alleles encoded as integers

    Notes
    -----
    If no allele is called at a position then a gap value (`-1`)
    is returned.

    """
    assert 0.5 < p <= 1.0 
    calls = np.zeros(array.shape[0:-1], dtype=dtype) -1
    indices = np.where(np.nan_to_num(array) >= p)
    calls[indices[0:-1]] = indices[-1]
    return calls


@numba.njit
def _sample_alleles(array, new):
    n_base, n_allele = array.shape
    for i in range(n_base):
        new[i] = -1  # default value is a gap

        for j in range(n_allele):
            val = array[i, j]

            if np.isnan(val):
                break

            else:
                new[i] = _random_choice(array[i])



def sample_alleles(array, dtype=np.int8):
    """Randomly sample an allele at each position based on
    the allele probabilities of each vector.

    Parameters
    ----------
    array : ndarray, float
        Array of row vectors encoding allele probabilities.
    dtype : dtype, optional
        Specify dtype of returned array.

    Returns
    -------
    array : int
        Array of alleles encoded as integers.

    """
    vector_size = array.shape[-1]
    probs = array.reshape(-1, vector_size)
    new = np.empty(len(probs), dtype=dtype)
    _sample_alleles(probs, new)
    new_shape = array.shape[0:-1]

    return new.reshape(new_shape)
