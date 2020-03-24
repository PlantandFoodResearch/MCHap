#!/usr/bin/env python3

import numpy as np
import numba


@numba.njit
def _as_probabilistic(array, new, n_alleles, probs, vector_size, gap_as_nan):

    for i in range(len(array)):
        a = array[i]  # allele index
        n = n_alleles[i]
        p = probs[i]

        # deal with gap
        if a == -1: # this is a gap
            if gap_as_nan:
                val = np.nan
            else:
                val = 1.0 / n
            for j in range(n):
                new[i, j] = val
            
            # pad with nan
            for j in range(n, vector_size):
                new[i, j] = np.nan # pad
        
        # deal with regular allele
        else:
            # TODO: find better way to do this?
            # not nesesarily a flat distribution over non-called alleles
            inv = (1.0 - p) / (n - 1) 
            for j in range(n):
                if j == a:
                    # called allele
                    new[i, j] = p
                else:
                    # non-called allele
                    new[i, j] = inv

            # pad with nan
            for j in range(n, vector_size):
                new[i, j] = np.nan 


def as_probabilistic(array, n_alleles, p=1.0, vector_size=None, gap_as_nan=True, dtype=np.float):

    if not isinstance(n_alleles, np.ndarray):
        n = n_alleles
        n_alleles = np.empty(array.shape, dtype=np.int)
        n_alleles[:] = [n]
    else:
        assert array.shape == n_alleles.shape

    if not isinstance(p, np.ndarray):
        p_scalar = p
        p = np.empty(array.shape, dtype=np.float)
        p[:] = [p_scalar]
    else:
        assert array.shape == p.shape

    if vector_size is None:
        vector_size = np.max(n_alleles)


    n_base = np.prod(array.shape)
    new = np.empty((n_base, vector_size), dtype=dtype)


    _as_probabilistic(
        array.ravel(),
        new,
        n_alleles.ravel(),
        p.ravel(),
        vector_size,
        gap_as_nan,
    )

    shape = array.shape + (vector_size, )

    return new.reshape(shape)

    
    
