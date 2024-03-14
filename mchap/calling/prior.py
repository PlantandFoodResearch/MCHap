import numpy as np
import numba
from math import lgamma

from mchap.jitutils import ln_equivalent_permutations
from mchap.calling.utils import count_allele, allelic_dosage


# TODO: consider pre-calculating alpha(s) at MCMC-model initialization
@numba.njit(cache=True)
def calculate_alphas(inbreeding, frequencies):
    """Calculate dispersion parameter of a Dirichlet-multinomial
    distribution assuming equal population frequency of each haplotype.

    Parameters
    ----------
    inbreeding : float
        Expected inbreeding coefficient of the sample.
    frequencies : ndarray, float, shape (n_alleles)
        Prior allele frequencies.

    Returns
    -------
    alphas : ndarray, float, shape (n_alleles)
        Dispersion parameters for each allele.
    """
    return frequencies * ((1 - inbreeding) / inbreeding)


@numba.njit(cache=True)
def log_genotype_allele_prior(
    genotype, variable_allele, unique_haplotypes, inbreeding=0, frequencies=None
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
    frequencies : ndarray, float, shape (n_alleles)
        Prior allele frequencies.

    Returns
    -------
    lprior : float
        Log-probability of the variable allele given the observed alleles.

    """
    assert 0 <= inbreeding < 1

    # if not inbred use null prior of a randomly chosen allele
    if inbreeding == 0:
        if frequencies is None:
            return np.log(1 / unique_haplotypes)
        else:
            return np.log(frequencies[genotype[variable_allele]])

    # count of alleles held as constant
    constant_sum = len(genotype) - 1

    # count of alleles held as constant that are IBS with variable allele
    constant_ibs = count_allele(genotype, genotype[variable_allele]) - 1

    # sum of alpha parameters accounting for constant alleles
    if frequencies is None:
        # base alpha for flat prior
        alpha = calculate_alphas(inbreeding, 1 / unique_haplotypes)
        sum_alpha = constant_sum + alpha * unique_haplotypes
        variable_alpha = alpha + constant_ibs
    else:
        alphas = calculate_alphas(inbreeding, frequencies)
        sum_alpha = constant_sum + alphas.sum()
        variable_alpha = alphas[genotype[variable_allele]] + constant_ibs

    # dirichlet-multinomial PMF
    left = lgamma(sum_alpha) - lgamma(1 + sum_alpha)
    right = lgamma(1 + variable_alpha) - lgamma(variable_alpha)
    return left + right


@numba.njit(cache=True)
def log_genotype_prior(genotype, unique_haplotypes, inbreeding=0, frequencies=None):
    """Prior probability of an individual genotype

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, )
        Integer encoded alleles in the proposed genotype.
    unique_haplotypes : int
        Number of possible haplotype alleles at this locus.
    inbreeding : float
        Expected inbreeding coefficient of the sample.
    frequencies : ndarray, float, shape (n_alleles)
        Prior allele frequencies.

    Returns
    -------
    lprior : float
        Log-prior probability of genotype.

    """
    assert 0 <= inbreeding < 1
    ploidy = len(genotype)
    dosage = allelic_dosage(genotype)

    # if not inbred use null prior
    if inbreeding == 0:
        ln_perms = ln_equivalent_permutations(dosage)
        if frequencies is None:
            return ln_perms - ploidy * np.log(unique_haplotypes)
        else:
            prod = 1
            for i in range(ploidy):
                prod *= frequencies[genotype[i]]
            return ln_perms + np.log(prod)

    # calculate the dispersion parameter for the PMF
    if frequencies is None:
        alpha_const = calculate_alphas(inbreeding, 1 / unique_haplotypes)
        sum_alphas = alpha_const * unique_haplotypes
    else:
        alphas = calculate_alphas(inbreeding, frequencies)
        sum_alphas = alphas.sum()

    # left side of equation in log space
    num = lgamma(ploidy + 1) + lgamma(sum_alphas)
    denom = lgamma(ploidy + sum_alphas)
    left = num - denom

    # right side of equation
    prod = 0.0  # log(1.0)
    for i in range(ploidy):
        dose = dosage[i]
        if dose > 0:
            if frequencies is None:
                alpha_i = alpha_const
            else:
                alpha_i = alphas[genotype[i]]
            num = lgamma(dose + alpha_i)
            denom = lgamma(dose + 1) + lgamma(alpha_i)
            prod += num - denom

    # return as log probability
    return left + prod
