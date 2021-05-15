#!/usr/bin/env python3

import numpy as np
import numba
from math import lgamma

from mchap.assemble import util
from mchap.assemble import arraymap

__all__ = [
    "log_likelihood",
    "log_likelihood_structural_change",
    "log_genotype_prior",
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

    intvl = util.interval_as_range(interval, n_base)

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
def log_likelihood_cached(reads, genotype, cache, read_counts=None):
    """Log likelihood of observed reads given a genotype with caching.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of
        probabilistic matrices.
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded
        as simple integers from 0 to n_nucl.
    cache : tuple
        An array_map tuple created with `new_log_likelihood_cache`.
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.

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
    cache,
    interval=None,
    read_counts=None,
):
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
    util.structural_change(
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


@numba.njit(cache=True)
def log_genotype_null_prior(dosage, unique_haplotypes):
    """Prior probability of a dosage for a non-inbred individual
    assuming all haplotypes are equally probable.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy, )
        Haplotype dosages within a genotype.
    unique_haplotypes : int
        Total number of unique possible haplotypes.

    Returns
    -------
    lprior : float
        Log-prior probability of dosage.

    """
    ploidy = dosage.sum()
    genotype_perms = util.count_equivalent_permutations(dosage)
    log_total_perms = ploidy * np.log(unique_haplotypes)
    return np.log(genotype_perms) - log_total_perms


@numba.njit(cache=True)
def _log_dirichlet_multinomial_pmf(dosage, dispersion, unique_haplotypes):
    """Dirichlet-Multinomial probability mass function assuming all categories
    (haplotypes) have equal dispersion parameters (alphas).

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy, )
        Counts of the observed haplotypes.
    dispersion : float
        Dispersion parameter for every possible haplotype.
    unique_haplotypes : int
        Total number of unique possible haplotypes.

    Returns
    -------
    lprob : float
        Log-probability.

    """
    ploidy = np.sum(dosage)
    sum_dispersion = dispersion * unique_haplotypes

    # left side of equation in log space
    num = np.log(util.factorial_20(ploidy)) + lgamma(sum_dispersion)
    denom = lgamma(ploidy + sum_dispersion)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(len(dosage)):
        dose = dosage[i]
        if dose > 0:
            num = lgamma(dose + dispersion)
            denom = np.log(util.factorial_20(dose)) + lgamma(dispersion)
            prod += num - denom

    # return as log probability
    return left + prod


@numba.njit(cache=True)
def log_genotype_prior(dosage, unique_haplotypes, inbreeding=0):
    """Prior probability of a dosage for an individual genotype
    assuming all haplotypes are equally probable.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy, )
        Haplotype dosages within a genotype.
    unique_haplotypes : int
        Total number of unique possible haplotypes.
    inbreeding : float
        Expected inbreeding coefficient in the interval (0, 1).

    Returns
    -------
    lprior : float
        Log-prior probability of dosage.

    """
    assert 0 <= inbreeding < 1

    # if not inbred use null prior
    if inbreeding == 0:
        return log_genotype_null_prior(dosage, unique_haplotypes)

    # calculate the dispersion parameter for the PMF
    dispersion = (1 / unique_haplotypes) * ((1 - inbreeding) / inbreeding)

    # calculate log-prior from Dirichlet-Multinomial PMF
    return _log_dirichlet_multinomial_pmf(dosage, dispersion, unique_haplotypes)
