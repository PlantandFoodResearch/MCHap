#!/usr/bin/env python3

import numpy as np
import numba

from mchap.assemble import util

__all__ = [
    "log_likelihood",
    "log_likelihood_structural_change",
    "log_genotype_prior",
]


@numba.njit
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


@numba.njit
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


@numba.njit
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


@numba.njit
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
    num = np.log(util.factorial_20(ploidy)) + util.log_gamma(sum_dispersion)
    denom = util.log_gamma(ploidy + sum_dispersion)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(len(dosage)):
        dose = dosage[i]
        if dose > 0:
            num = util.log_gamma(dose + dispersion)
            denom = np.log(util.factorial_20(dose)) + util.log_gamma(dispersion)
            prod += num - denom

    # return as log probability
    return left + prod


@numba.njit
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
