#!/usr/bin/env python3

import numpy as np
import numba


@numba.njit
def _as_probabilistic(array, new, n_alleles, probs, vector_size, gaps, pad):

    for i in range(len(array)):
        a = array[i]  # allele index
        n = n_alleles[i]
        p = probs[i]

        # deal with gap
        if a == -1: # this is a gap
            if gaps:
                new[i] = np.nan  # fill vector with nan
            else:
                val = 1.0 / n
                for j in range(n):
                    new[i, j] = val
                
                # pad distribution
                for j in range(n, vector_size):
                    new[i, j] = pad # pad
        
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

            # pad
            for j in range(n, vector_size):
                new[i, j] = pad 


def as_probabilistic(array, 
                     n_alleles, 
                     p=1.0, 
                     vector_size=None, 
                     gaps=True, 
                     nan_pad=False,
                     dtype=np.float):

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

    # determine pad type
    if nan_pad:
        pad = np.nan
    else:
        pad = 0.0

    _as_probabilistic(
        array.ravel(),
        new,
        n_alleles.ravel(),
        p.ravel(),
        vector_size,
        gaps,
        pad
    )

    shape = array.shape + (vector_size, )

    return new.reshape(shape)

    
def vector_from_string(string, gaps='-', length=None, dtype=np.int8):
    if length is None:
        length = len(string)

    # gap is default
    vector = np.zeros(length, dtype=dtype) - 1

    for i in range(min(length, len(string))):
        char = string[i]
        if char in gaps:
            vector[i] = -1
        else:
            vector[i] = int(char)
    return vector


def from_strings(data, gaps='-', length=None, dtype=np.int8):
    if isinstance(data, str):
        return vector_from_string(data, gaps=gaps, length=length, dtype=dtype)
    
    if isinstance(data, np.ndarray):
        pass
    else:
        data = np.array(data, copy=False)

    sequences = data.ravel()

    # default to length of longest element
    if length is None:
        length =  max(len(i) for i in sequences)

    # number of sequences
    n_seq = len(sequences)

    # new array with gap as default
    array = np.empty((n_seq, length), dtype=dtype)

    for i in range(n_seq):
        array[i] = vector_from_string(
            sequences[i],
            gaps=gaps,
            length=length,
            dtype=dtype
        )

    shape = data.shape + (length, )

    return array.reshape(shape)


def vector_as_string(vector, gap='-'):
    return ''.join(str(i) if i >= 0 else gap for i in vector)


def as_strings(array, gap='-'):
    if not isinstance(array, np.ndarray):
        array = np.array(array, copy=False)
    if array.ndim == 1:
        return vector_as_string(array, gap=gap)

    shape = array.shape[:-1]
    length = array.shape[-1]
    dtype = 'U{}'.format(length)
    n_seq = np.prod(shape)
    vectors = array.reshape(n_seq, length)

    strings = np.empty(n_seq, dtype=dtype)

    for i in range(n_seq):
        strings[i] = vector_as_string(vectors[i], gap=gap)

    return strings.reshape(shape)




    
