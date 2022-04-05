#!/usr/bin/env python3

import numpy as np
import numba
from math import lgamma

from mchap.jitutils import ln_equivalent_permutations


__all__ = [
    "log_genotype_prior",
]


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
    ln_perms = ln_equivalent_permutations(dosage)
    ln_total_perms = ploidy * np.log(unique_haplotypes)
    return ln_perms - ln_total_perms


@numba.njit(cache=True)
def log_dirichlet_multinomial_pmf(dosage, dispersion, unique_haplotypes):
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
    num = lgamma(ploidy + 1) + lgamma(sum_dispersion)
    denom = lgamma(ploidy + sum_dispersion)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(len(dosage)):
        dose = dosage[i]
        if dose > 0:
            num = lgamma(dose + dispersion)
            denom = lgamma(dose + 1) + lgamma(dispersion)
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
    return log_dirichlet_multinomial_pmf(dosage, dispersion, unique_haplotypes)
