import numpy as np
import numba

from mchap.assemble.util import factorial_20
from mchap.assemble.likelihood import lgamma


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
def log_dirichlet_multinomial_pmf(allele_counts, dispersion):
    """Dirichlet-Multinomial probability mass function.

    Parameters
    ----------
    allele_counts : ndarray, int, shape (n_alleles, )
        Counts of possible alleles.
    dispersion : ndarray, float, shape (n_alleles, )
        Dispersion parameters (alphas) for each possible allele.

    Returns
    -------
    lprob : float
        Log-probability.

    """
    assert allele_counts.shape == dispersion.shape
    sum_counts = np.sum(allele_counts)
    sum_dispersion = dispersion.sum()

    # left side of equation in log space
    num = np.log(factorial_20(sum_counts)) + lgamma(sum_dispersion)
    denom = lgamma(sum_counts + sum_dispersion)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(len(allele_counts)):
        count = allele_counts[i]
        disp = dispersion[i]
        if count > 0:
            num = lgamma(count + disp)
            denom = np.log(factorial_20(count)) + lgamma(disp)
            prod += num - denom

    # return as log probability
    return left + prod


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

    # calculate the dispersion parameters for the PMF
    # this is the default value based on the inbreeding coefficient
    # which is then updated base on the observed 'constant' alleles
    default_dispersion = inbreeding_as_dispersion(inbreeding, unique_haplotypes)
    dispersion = np.full(unique_haplotypes, default_dispersion)
    for i in range(len(genotype)):
        if i != variable_allele:
            a = genotype[i]
            dispersion[a] += 1

    # indicate the variable allele as a one-hot array
    allele_counts = np.zeros(dispersion.shape, np.int64)
    a = genotype[variable_allele]
    allele_counts[a] += 1

    # calculate log-prior from Dirichlet-Multinomial PMF
    return log_dirichlet_multinomial_pmf(allele_counts, dispersion)
