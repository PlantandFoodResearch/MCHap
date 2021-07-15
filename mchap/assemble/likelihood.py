#!/usr/bin/env python3

import numpy as np
import numba

from mchap.jitutils import (
    structural_change,
)
from mchap.assemble import arraymap

__all__ = [
    "log_likelihood",
    "log_likelihood_structural_change",
]


@numba.njit(cache=True)
def log_likelihood(reads, genotype, read_counts=None):
    """Log likelihood of observed reads given a genotype.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of
        probabilistic matrices.
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded
        as simple integers from 0 to n_nucl.
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype.

    """

    ploidy, n_base = genotype.shape
    n_reads = len(reads)

    llk = 0.0

    for r in range(n_reads):

        read_prob = 0

        for h in range(ploidy):
            read_hap_prod = 1.0

            for j in range(n_base):
                i = genotype[h, j]

                val = reads[r, j, i]

                if np.isnan(val):
                    pass
                else:
                    read_hap_prod *= val

            read_prob += read_hap_prod / ploidy

        log_read_prob = np.log(read_prob)

        if read_counts is not None:
            log_read_prob *= read_counts[r]

        llk += log_read_prob

    return llk


@numba.njit(cache=True)
def log_likelihood_structural_change(
    reads, genotype, haplotype_indices, interval=None, read_counts=None
):
    """Log likelihood of observed reads given a genotype given a structural change.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of
        probabilistic matrices.
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as
        simple integers from 0 to n_nucl.
    haplotype_indices : ndarray, int, shape (ploidy)
        Indicies of haplotypes to use within the
        changed interval.
    interval : tuple, int, shape (2, ), optional
        Interval of base-positions to swap (defaults to
        all base positions).
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the proposed
        structural change to the genotype.

    """
    ploidy, n_base = genotype.shape
    n_reads = len(reads)

    if interval is None:
        intvl = range(n_base)
    else:
        intvl = range(interval[0], interval[1])

    llk = 0.0

    for r in range(n_reads):

        read_prob = 0

        for h in range(ploidy):
            read_hap_prod = 1.0

            for j in range(n_base):

                # check if in the altered region
                if j in intvl:
                    # use base from alternate hap
                    h_ = haplotype_indices[h]
                else:
                    # use base from current hap
                    h_ = h

                # get nucleotide index
                i = genotype[h_, j]

                val = reads[r, j, i]

                if np.isnan(val):
                    pass
                else:
                    read_hap_prod *= val

            read_prob += read_hap_prod / ploidy

        log_read_prob = np.log(read_prob)

        if read_counts is not None:
            log_read_prob *= read_counts[r]

        llk += log_read_prob

    return llk


@numba.njit(cache=True)
def new_log_likelihood_cache(ploidy, n_base, max_alleles, max_size=2 ** 16):
    """Create an array_map forcaching log-likelihoods.

    Parameters
    ----------
    ploidy : int
        Ploidy of genotype.
    n_base : int
        Number of base postions in genotype.
    max_alleles : int
        Maximum number of alleles at any given postion in genotype.
    max_size : int
        Maximum array length for nodes and values at which point storing
        new values will raise an error.

    Returns
    -------
    tree : np.array, int, shape (initial_size, node_branches)
        Array of array_map nodes in which -1 in a null value.
    values : nd.array, float, shape (initial_size, )
        Array of values stored in array_map.
    array_length : int
        Fixed size of arrays stored in this array_map.
    empty_node : int
        Index of the first empty node slot excluding 0.
    empty_value : int
        Index of the first empty values slot.
    max_size : int
        Maximum array length for nodes and values at which point storing
        new values will raise an error.

    Notes
    -----
    Returned values are not meant to be interacted with individually and should be
    treated as a single object.

    """
    return arraymap.new(
        ploidy * n_base, max_alleles, initial_size=64, max_size=max_size
    )


@numba.njit(cache=True)
def log_likelihood_cached(reads, genotype, read_counts=None, cache=None):
    """Log likelihood of observed reads given a genotype with caching.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of
        probabilistic matrices.
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded
        as simple integers from 0 to n_nucl.
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.
    cache : tuple
        An array_map tuple created with `new_log_likelihood_cache`.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype.
    cache : tuple
        An updated array_map tuple.

    Notes
    -----
    Elements of the cache array_map may be updated in place or replaced
    hence existing references to the array_map should not be reused.
    """
    if cache is None:
        llk = log_likelihood(reads, genotype, read_counts=read_counts)
        return llk, cache

    # try retrive from cache
    llk = arraymap.get(cache, genotype.ravel())
    if np.isnan(llk):
        # calculate and update cache
        llk = log_likelihood(reads, genotype, read_counts=read_counts)
        # the cache will emptied if it is full
        cache = arraymap.set(cache, genotype.ravel(), llk, empty_if_full=True)
    return llk, cache


@numba.njit(cache=True)
def log_likelihood_structural_change_cached(
    reads,
    genotype,
    haplotype_indices,
    interval=None,
    read_counts=None,
    cache=None,
):
    """Log likelihood of observed reads given a genotype given a structural change.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of
        probabilistic matrices.
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as
        simple integers from 0 to n_nucl.
    haplotype_indices : ndarray, int, shape (ploidy)
        Indicies of haplotypes to use within the
        changed interval.
    interval : tuple, int, shape (2, ), optional
        Interval of base-positions to swap (defaults to
        all base positions).
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.
    cache : tuple
        An array_map tuple created with `new_log_likelihood_cache`.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the proposed
        structural change to the genotype.
    cache : tuple
        An updated array_map tuple.

    """
    if cache is None:
        llk = log_likelihood_structural_change(
            reads=reads,
            genotype=genotype,
            haplotype_indices=haplotype_indices,
            interval=interval,
            read_counts=read_counts,
        )
        return llk, cache

    # try retrive from cache
    genotype_new = genotype.copy()
    structural_change(
        genotype_new, haplotype_indices=haplotype_indices, interval=interval
    )
    llk = arraymap.get(cache, genotype_new.ravel())
    if np.isnan(llk):
        # calculate and update cache
        llk = log_likelihood_structural_change(
            reads=reads,
            genotype=genotype,
            haplotype_indices=haplotype_indices,
            interval=interval,
            read_counts=read_counts,
        )
        # the cache will emptied if it is full
        cache = arraymap.set(cache, genotype_new.ravel(), llk, empty_if_full=True)
    return llk, cache
