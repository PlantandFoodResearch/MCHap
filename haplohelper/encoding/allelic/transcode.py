#!/usr/bin/env python3

import numpy as np
import numba


@numba.njit
def _as_probabilistic(array, new, n_alleles, probs, vector_size, gaps):

    for i in range(len(array)):
        a = array[i]  # allele index
        n = n_alleles[i]
        p = probs[i]

        # deal with gap
        if a == -1: # this is a gap
            if gaps:
                val = np.nan
            else:
                val = 1.0 / n
            for j in range(n):
                new[i, j] = val
            
            # always pad distribution with 0, even a gap
            for j in range(n, vector_size):
                new[i, j] = 0.0 # pad
        
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
                new[i, j] = 0.0 


def _tile_to_shape(x, shape):
    ndims = np.ndim(x)
    diff = len(shape) - ndims
    assert np.shape(x) == shape[diff:]
    template = shape[:diff] + tuple(1 for _ in range(ndims))
    return np.tile(x, template)


def as_probabilistic(array, 
                     n_alleles, 
                     p=1.0, 
                     gaps=True, 
                     dtype=np.float):

    vector_size = np.max(n_alleles)

    n_alleles = _tile_to_shape(n_alleles, array.shape)
    p = _tile_to_shape(p, array.shape)

    n_base = np.prod(array.shape)
    new = np.empty((n_base, vector_size), dtype=dtype)

    _as_probabilistic(
        array.ravel(),
        new,
        n_alleles.ravel(),
        p.ravel(),
        vector_size,
        gaps
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


def vector_as_string(vector, gap='-', alleles=None):
    if alleles is None:
        return ''.join(str(a) if a >= 0 else gap for a in vector)
    else:
        return ''.join(alleles[i][a] if a >= 0 else gap for i, a in enumerate(vector))


def as_strings(array, gap='-', alleles=None):
    if not isinstance(array, np.ndarray):
        array = np.array(array, copy=False)
    if array.ndim == 1:
        return vector_as_string(array, gap=gap, alleles=alleles)

    shape = array.shape[:-1]
    length = array.shape[-1]
    dtype = 'U{}'.format(length)
    n_seq = np.prod(shape)
    vectors = array.reshape(n_seq, length)

    strings = np.empty(n_seq, dtype=dtype)

    for i in range(n_seq):
        strings[i] = vector_as_string(vectors[i], gap=gap, alleles=alleles)

    return strings.reshape(shape)


def vector_as_characters(vector, gap='-', alleles=None):
    if alleles is None:
        return np.fromiter(
            (str(a) if a >= 0 else gap for a in vector), 
            dtype='U1', 
            count=len(vector)
        )
    else:
        return np.fromiter((alleles[i][a] if a >= 0 else gap for i, a in enumerate(vector)), 
            dtype='U1', 
            count=len(vector)
        )

    
def as_characters(array, gap='-', alleles=None):

    if not isinstance(array, np.ndarray):
        array = np.array(array, copy=False)
    if array.ndim == 1:
        return vector_as_characters(array, gap=gap, alleles=alleles)

    shape = array.shape
    n_seq = np.prod(shape[:-1])
    vectors = array.reshape(n_seq, -1)

    chars = np.empty(vectors.shape, dtype='U1')

    for i in range(n_seq):
        chars[i] = vector_as_characters(vectors[i], gap=gap, alleles=alleles)

    return chars.reshape(shape)
