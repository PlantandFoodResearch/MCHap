#!/usr/bin/env python3

import numpy as np


def as_probabilistic(
    array, 
    n_alleles=4, 
    p=1.0, 
    error_factor=3, 
    dtype=float):
    """Converts an array of integer encoded alleles to an 
    array of probabilistic row vectors.

    Parameters
    ----------
    array : ndarray, int
        Array of integers encoding alleles
    n_alleles : array_like, int
        Constrain number of alleles encoded at each position.
    p : array_like, float, optional
        Probability that allele call is correct.
    error_factor:
        Factor to divide error rate by for probability of each
        alt alleles, for a SNP this should be 3 (default = 3).
    dtype : dtype
        Specify the dtype of the returned probabilistic array.

    Returns
    -------
    probabilistic_array : ndarray, float, shape(n_positions, max_allele)
        Array of vectors encoding allele probabilities.

    Notes
    -----
    In the case of an incorrect call it is assumed that each of the 
    non-called alleles are equally likely to be called.
    In this case the error rate (1 - p) is divided by the error_factor
    (the number of possible non-called alleles) to get the probability of 
    each of the non-called alleles.
    If an position is constrained to have fewer than the possible number 
    of alternate alleles (e.g. a bi-allelic constraint) using the
    n_alleles argument then the probability across all remaining alleles
    will sum to less than 1.    
    """
    # check inputs
    array = np.array(array, copy=False)
    n_alleles = np.array(n_alleles, copy=False)
    error_factor = np.array(error_factor, copy=False)
    p = np.array(p, copy=False)
    
    # special case for zero-length reads
    if array.shape[-1] == 0:
        return np.empty(array.shape + (0,), dtype=dtype)
    
    alleles = np.arange(np.max(n_alleles))

    # onehot encoding of alleles
    onehot = array[..., None] == alleles
    
    # basic probs
    new = ((1 - p) / error_factor)[..., None] * ~onehot
    calls = p[..., None] * onehot
    new[onehot] = calls[onehot]

    # nan fill gaps
    new[array < 0] = np.nan

    # zero out non-alleles
    new[..., n_alleles[..., None] <= alleles] = 0

    return new

    
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
