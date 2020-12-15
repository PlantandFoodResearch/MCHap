import numpy as np 
import numba

from mchap.assemble import util


@numba.njit
def genotype_prior(genotype, unique_haplotypes):
    ploidy, _ = genotype.shape
    dosage = np.zeros(ploidy, dtype=np.int8)
    util.get_dosage(dosage, genotype)
    n_perms = util.count_equivalent_permutations(dosage)
    total_perms = unique_haplotypes ** ploidy
    return n_perms / total_perms


@numba.njit
def chain_swap_acceptance(
    genotype_i,
    llk_i,
    prior_i,
    temp_i,
    genotype_j,
    llk_j,
    prior_j,
    temp_j,
):
    lk_i = np.exp(llk_i)
    lk_j = np.exp(llk_j)

    unnormalized_posterior_i = lk_i * prior_i
    unnormalized_posterior_j = lk_j * prior_j

    frac_1 = (unnormalized_posterior_j / unnormalized_posterior_i) ** temp_i
    frac_2 = (unnormalized_posterior_i / unnormalized_posterior_j) ** temp_j

    acceptance_ratio = frac_1 * frac_2
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
    prior_i = genotype_prior(genotype_i, unique_haplotypes)
    prior_j = genotype_prior(genotype_j, unique_haplotypes)

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
