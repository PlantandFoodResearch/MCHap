#!/usr/bin/env python3

import numpy as np
import numba

from haplohelper.assemble.step.util import random_choice


@numba.njit
def _call_alleles(array, new, p,):
    n_base, n_allele = array.shape
    for i in range(n_base):
        new[i] = -1  # default value is a gap

        for j in range(n_allele):
            val = array[i, j]

            if np.isnan(val):
                break

            elif val >= p:
                new[i] = j
                break

            else:
                # no alleles meet threshold
                pass


def call_alleles(array, p=0.95, dtype=np.int8):
    if not (0.5 < p <= 1.0):
        raise ValueError('p must be a probability in the interval 0.5 < p <= 1')

    vector_size = array.shape[-1]
    probs = array.reshape(-1, vector_size)
    new = np.empty(len(probs), dtype=dtype)
    _call_alleles(array, new, p)
    new_shape = array.shape[0:-1]

    return new.reshape(new_shape)


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
                new[i] = random_choice(array[i])



def sample_alleles(array, dtype=np.int8):
    """Returns a random sample of a probability vector using each element as a
    probability distribution.
    The encoding of each element is expected to be a probability distribution
    i.e. the sum of all values in the element is 1.
    A vector of 'one-hot' elements of equal length to the sampled
    vector is returned.
    For example, an element encoded `[0, 0, 0.5, 0.5]` will return
    `[0, 0, 1, 0]` or `[0, 0, 0, 1]` with probability 1/2  each.
    An element encoded `[0, 0, 0.666, 0.333]` will return  `[0, 0, 1, 0]` or
    `[0, 0, 0, 1]` with probability 2/3 or 1/3  respectively.
    A nan element `[nan, nan, nan, nan]` is interpreted as a gap in the
    sequence and will return the zero element `[0, 0, 0, 0]`.
    """
    vector_size = array.shape[-1]
    probs = array.reshape(-1, vector_size)
    new = np.empty(len(probs), dtype=dtype)
    _sample_alleles(array, new)
    new_shape = array.shape[0:-1]

    return new.reshape(new_shape)
