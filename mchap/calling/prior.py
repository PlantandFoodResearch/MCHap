import numpy as np
import numba
from math import lgamma

from mchap.calling.utils import allelic_dosage, count_allele


@numba.njit(cache=True)
def inbreeding_as_dispersion(inbreeding, unique_haplotypes):
    """Calculate dispersion parameter of a Dirichlet-multinomial
    distribution assuming equal population frequency of each haplotype.

    Parameters
    ----------
    inbreeding : float
        Expected inbreeding coefficient of the sample.
    unique_haplotypes : int
        Number of possible haplotype alleles at this locus.

    Returns
    -------
    dispersion : float
        Dispersion parameter for all haplotypes.
    """
    return (1 / unique_haplotypes) * ((1 - inbreeding) / inbreeding)


@numba.njit(cache=True)
def log_genotype_allele_prior(
    genotype, variable_allele, unique_haplotypes, inbreeding=0
):
    """Log probability that a genotype contains a specified allele
    given its other alleles are treated as constants.

    This prior function is designed to be used in a gibbs sampler in which a single allele
    is resampled at a time.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, )
        Integer encoded alleles in the proposed genotype.
    variable_allele : int
        Index of the allele that is not being held constant ( < ploidy).
    unique_haplotypes : int
        Number of possible haplotype alleles at this locus.
    inbreeding : float
        Expected inbreeding coefficient of the sample.

    Returns
    -------
    lprior : float
        Log-probability of the variable allele given the observed alleles.

    """
    assert 0 <= inbreeding < 1

    # if not inbred use null prior of a randomly chosen allele
    if inbreeding == 0:
        return np.log(1 / unique_haplotypes)

    # base alpha for flat prior
    alpha = inbreeding_as_dispersion(inbreeding, unique_haplotypes)

    # sum of alpha parameters accounting for constant alleles
    constant = np.delete(genotype, variable_allele)
    counts = allelic_dosage(constant)
    sum_alpha = np.sum(counts) + alpha * unique_haplotypes

    # alpha parameter for variable allele
    count = count_allele(genotype, genotype[variable_allele]) - 1
    variable_alpha = alpha + count

    # dirichlet-multinomial PMF
    left = lgamma(sum_alpha) - lgamma(1 + sum_alpha)
    right = lgamma(1 + variable_alpha) - lgamma(variable_alpha)
    return left + right
