import numpy as np 
import numba

from mchap.assemble import util


@numba.njit
def log_genotype_prior(genotype, unique_haplotypes):
    ploidy, _ = genotype.shape
    dosage = np.zeros(ploidy, dtype=np.int8)
    util.get_dosage(dosage, genotype)
    n_perms = util.count_equivalent_permutations(dosage)
    total_perms = unique_haplotypes ** ploidy
    return np.log(n_perms) - np.log(total_perms)


@numba.njit
def chain_swap_acceptance(
    genotype_i,
    llk_i,
    log_prior_i,
    temp_i,
    genotype_j,
    llk_j,
    log_prior_j,
    temp_j,
):

    unnormalized_posterior_i = llk_i + log_prior_i
    unnormalized_posterior_j = llk_j + log_prior_j

    frac_1 = (unnormalized_posterior_j - unnormalized_posterior_i) * temp_i
    frac_2 = (unnormalized_posterior_i - unnormalized_posterior_j) * temp_j

    acceptance_ratio = np.exp(frac_1 + frac_2)
    if acceptance_ratio > 1.0:
        acceptance_ratio = 1.0
    
    return acceptance_ratio


@numba.njit
def chain_swap_step(
    genotype_i,
    llk_i,
    temp_i,
    genotype_j,
    llk_j,
    temp_j,
    unique_haplotypes,
):
    prior_i = log_genotype_prior(genotype_i, unique_haplotypes)
    prior_j = log_genotype_prior(genotype_j, unique_haplotypes)

    acceptance = chain_swap_acceptance(
        genotype_i,
        llk_i,
        prior_i,
        temp_i,
        genotype_j,
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
