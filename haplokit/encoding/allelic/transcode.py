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
    """Converts an array integer encoded alleles to an 
    array of probabilistic row vectors.

    Parameters
    ----------
    array : ndarray, int
        Array of integers encoding alleles
    n_alleles : array_like, int
        Number of alleles to encode in each row vector.
    p : array_like, float, optional
        Probability associated with each allele call in the input array.
    gaps : bool, optional
        If `False` then gaps in the input array are treated as unknown
        values in the probabilistic array, i.e. equal probability of 
        each allele.
    dtype : dtype
        Specify the dtype of the returned probabilistic array.

    Returns
    -------
    probabilistic_array : ndarray, float, shape(n_positions, max_allele)
        Array of vectors encoding allele probabilities.

    Notes
    -----
    If n_alleles is variable then the vectors encoding fewer alleles
    will be padded with `0` values.
    If gaps are included they will be represented as a vector of `nan`
    values which will also be padded with `0` values if as required.
    
    """
    vector_size = int(np.max(n_alleles, initial=0))

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
    """Convert a string to an array of integer encoded alleles.

    Parameters
    ----------
    string : str
        String of alleles
    gaps : str, optional
        String of symbols to be interpreted as gaps in the sequence.
    length : int, optional
        Truncate or extend sequence to a set length by padding with gap values.
    dtype : dtype, optional
        Specify dtype of returned array.
    
    Returns
    -------
    array : ndarray, int
        Array of alleles encoded as integers.

    """
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
    """Convert a series of strings to an array of integer encoded alleles.

    Parameters
    ----------
    data : array_like, str
        Sequence of strings of alleles.
    gaps : str, optional
        String of symbols to be interpreted as gaps in the sequence.
    length : int, optional
        Truncate or extend sequence to a set length by padding with gap values.
    dtype : dtype, optional
        Specify dtype of returned array.
    
    Returns
    -------
    array : ndarray, int
        Array of alleles encoded as integers.

    """
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
    """Convert a vector of integer encoded alleles to a string.

    Parameters
    ----------
    vector : ndarray, int, (n_pos, )
        1D array of integers.
    gap : str, optional
        Character used to represent gap values.
    alleles : array_like, array_like, str, optional
        Characters used to represent each allele at each position.
    
    Returns
    -------
    string : str
        String of allele characters

    """
    if alleles is None:
        return ''.join(str(a) if a >= 0 else gap for a in vector)
    else:
        return ''.join(alleles[i][a] if a >= 0 else gap for i, a in enumerate(vector))


def as_strings(array, gap='-', alleles=None):
    """Convert an array of integer encoded alleles into one or more strings.

    Parameters
    ----------
    array : ndarray, int
        array of integers.
    gap : str, optional
        Character used to represent gap values.
    alleles : array_like, array_like, str, optional
        Characters used to represent each allele at each position.
    
    Returns
    -------
    strings : ndarray, str
        Strings of allele characters

    """
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
    """Convert an array of integer encoded alleles into an array of characters.

    Parameters
    ----------
    vector : ndarray, int, (n_pos, )
        1D array of integers.
    gap : str, optional
        Character used to represent gap values.
    alleles : array_like, array_like, str, optional
        Characters used to represent each allele at each position.
    
    Returns
    -------
    Characters : ndarray, str, (n_pos, )
        1D array of allele characters.

    """
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
    """Convert an array of integer encoded alleles into an array of characters.

    Parameters
    ----------
    array : ndarray, int
        An array of integers.
    gap : str, optional
        Character used to represent gap values.
    alleles : array_like, array_like, str, optional
        Characters used to represent each allele at each position.
    
    Returns
    -------
    characters : ndarray, str
        Array of allele characters.

    """

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
