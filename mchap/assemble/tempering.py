import numpy as np
import numba

from mchap.jitutils import get_haplotype_dosage
from mchap.assemble.prior import log_genotype_prior

__all__ = ["chain_swap_step"]


@numba.njit(cache=True)
def chain_swap_acceptance(
    llk_i,
    log_prior_i,
    temp_i,
    llk_j,
    log_prior_j,
    temp_j,
):
    """Acceptance probability for switching genotypes between chains
    of different temperatures.

    Parameters
    ----------
    llk_i : float
        Log likelihood for state in the cooler chain.
    log_prior_i : float
        Log prior probability for state in the cooler chain.
    temp_i : float
        Inverse temperature of the cooler chain.
    llk_j : float
        Log likelihood for state in the warmer chain.
    log_prior_j : float
        Log prior probability for state in the warmer chain.
    temp_j : float
        Inverse temperature of the warmer chain.

    Returns
    -------
    acceptance_ratio : float
        Probability of accepting a state exchange.

    Notes
    -----
    Calculation following equation 11 of Sambridge (2014).
    """
    assert temp_i > temp_j

    unnormalized_posterior_i = llk_i + log_prior_i
    unnormalized_posterior_j = llk_j + log_prior_j

    frac_1 = (unnormalized_posterior_j - unnormalized_posterior_i) * temp_i
    frac_2 = (unnormalized_posterior_i - unnormalized_posterior_j) * temp_j

    acceptance_ratio = np.exp(frac_1 + frac_2)
    if acceptance_ratio > 1.0:
        acceptance_ratio = 1.0

    return acceptance_ratio


@numba.njit(cache=True)
def chain_swap_step(
    genotype_i,
    llk_i,
    temp_i,
    genotype_j,
    llk_j,
    temp_j,
    unique_haplotypes,
    inbreeding=0,
):
    """Exchange-swap step for exchanging genotypes between chains
    of different temperatures.

    Parameters
    ----------
    genotype_i : float
        Genotype state of the cooler chain.
    llk_i : float
        Log likelihood for state in the cooler chain.
    temp_i : float
        Inverse temperature of the cooler chain.
    genotype_j : float
        Genotype state of the warmer chain.
    llk_j : float
        Log likelihood for state in the warmer chain.
    temp_j : float
        Inverse temperature of the warmer chain.
    unique_haplotypes : int
        Number of possible unique haplotypes in both chains.
    inbreeding : float
        Expected inbreeding coefficient of organism.

    Returns
    -------
    llk_i : float
        Updated log likelihood for state in the cooler chain.
    llk_j : float
        Updated log likelihood for state in the warmer chain.

    Notes
    -----
    If a step is made then genotypes are modified in place.

    """
    ploidy, _ = genotype_i.shape
    dosage = np.zeros(ploidy, dtype=np.int8)

    # prior for genotype i dosage
    get_haplotype_dosage(dosage, genotype_i)
    prior_i = log_genotype_prior(dosage, unique_haplotypes, inbreeding=inbreeding)

    # prior for genotype j dosage
    get_haplotype_dosage(dosage, genotype_j)
    prior_j = log_genotype_prior(dosage, unique_haplotypes, inbreeding=inbreeding)

    acceptance = chain_swap_acceptance(
        llk_i,
        prior_i,
        temp_i,
        llk_j,
        prior_j,
        temp_j,
    )

    val = np.random.rand()
    if acceptance >= val:
        # swap chains
        genotype_i_new = genotype_j.copy()
        genotype_j_new = genotype_i.copy()
        genotype_i[:] = genotype_i_new
        genotype_j[:] = genotype_j_new
        return llk_j, llk_i
    else:
        # no swap
        return llk_i, llk_j
