#!/usr/bin/env python3

import numpy as np
from . import sequence


def iter_kmers(array, k=3):
    """Return a generator of aligned kmers given a array 
    of binary or onehot encoded sequences.
    """
    n_base = array.shape[-1]
    n_windows = n_base - (k - 1)
    masks = np.zeros((n_windows, n_base), dtype=np.bool)
    for i in range(n_windows):
        masks[i][i:i+k]=True
    for read in array.reshape(-1, n_base):
        for mask in masks:
            if np.any(sequence.is_gap(read[mask])):
                pass
            else:
                kmer = np.zeros((n_base), dtype=array.dtype) - 1
                kmer[mask] = read[mask]
                yield kmer


def kmer_counts(array, k=3, order=None):
    """Return a tuple of unique aligned kmers and counts 
    given a array of binary or onehot encoded sequences.
    """
    assert order in {'ascending', 'descending', None}
    kmer = None  # handle case of no kmers
    kmers_dict = {}
    counts_dict = {}
    for kmer in iter_kmers(array, k=k):
        string = kmer.tostring()
        if string not in kmers_dict:
            kmers_dict[string] = kmer
            counts_dict[string] = 1
        else:
            counts_dict[string] += 1
            
    if kmer is None:  # handle case of no kmers
        return np.array([], dtype=array.dtype), np.array([], dtype=np.int)

    n_base = len(kmer)
    n_kmer = len(kmers_dict)
    kmers = np.empty((n_kmer, n_base), dtype=array.dtype)
    counts = np.empty(n_kmer, dtype=np.int)

    for i, (string, kmer) in enumerate(kmers_dict.items()):
        kmers[i]=kmer
        counts[i]=counts_dict[string]

    if order is None:
        return kmers, counts

    idx = np.argsort(counts)
    if order == 'descending':
        idx = np.flip(idx, axis=0)

    return kmers[idx], counts[idx]


def kmer_positions(kmers, end=False):
    """Return the local alignment positions of bases
    for each kmer in an array of kmers.
    """
    assert end in {False, 'start', 'stop'}
    is_coding = ~sequence.is_gap(kmers)
    # detect k
    k = np.sum(is_coding, axis=-1)
    assert np.all(k[0] == k)
    k = k[0]
    if end == 'start':
        return np.where(is_coding)[1][0::k]
    elif end == 'stop':
        return np.where(is_coding)[1][k-1::k]
    else:
        return np.where(is_coding)[1].reshape(-1, k)


def kmer_frequency(kmers, counts):
    """Frequency of each kmer among kmers that share its position.
    """
    is_coding = ~sequence.is_gap(kmers)
    # detect k
    k = np.sum(is_coding, axis=-1)
    assert np.all(k[0] == k)
    k = k[0]
    # position of first coding base
    positions = np.where(is_coding)[1][0::k]
    # number of kmers starting at each position
    depths = np.zeros(kmers.shape[-2] - (k-1), dtype=np.int)
    for i, pos in enumerate(positions):
        depths[pos] += counts[i]
    # kmer frequency at it's position
    freqs = np.empty(len(kmers), dtype=np.float)
    for i, pos in enumerate(positions):
        freqs[i] = counts[i] / depths[pos]
    return freqs
