#!/usr/bin/env python3

import numpy as np 
import numba

from mchap.assemble import util
from mchap.assemble.likelihood import log_likelihood


@numba.njit
def base_step(genotype, reads, llk, h, j, mask=None, temp=1):
    """Mutation Gibbs sampler step for the jth base position 
    of the hth haplotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded 
        as simple integers from 0 to n_nucl.
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic 
        matrices.
    llk : float
        Log-likelihood of the initial haplotype state given 
        the observed reads.
    h : int
        Index of the haplotype to be mutated.
    j : int
        Index of the base position to be mutated
    mask : ndarray, bool, shape (n_nucl, )
        Optionally indicate some alleles to skip e.g. alleles 
        which are known to have 0 probability.
    temp : float
        An inverse temperature in the interval 0, 1 to adjust
        the sampled distribution by.
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given 
        the updated genotype.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    assert  0 <= temp <= 1
    # number of possible alleles given given array size
    n_alleles = reads.shape[-1]

    # store log likelihoods calculated with each allele
    llks = np.empty(n_alleles)

    # store MH acceptance probability for each allele
    log_accept = np.empty(n_alleles)
    
    # use differences in dosage of haplotype h to calculate
    # ratio of priors and ratio of proposal probabilities
    lhapcount = np.log(util.count_haplotype_copies(genotype, h))

    current_nucleotide = genotype[h, j]    
    for i in range(n_alleles):
        if (mask is not None) and mask[i]:
            # exclude masked allele
            llks[i] = - np.inf  # log(0.0)
            log_accept[i] = - np.inf
        else:
            if i == current_nucleotide:
                # store current likelihood
                llks[i] = llk
                log_accept[i] = 0.0  # log(1.0)
            else:
                genotype[h, j] = i
                llk_i = log_likelihood(reads, genotype)
                lhapcount_i = np.log(util.count_haplotype_copies(genotype, h))
                llks[i] = llk_i
                llk_ratio = llk_i - llk
                # ratio of priors and ratio of proposal probs are each others inverse
                lprior_ratio = lhapcount - lhapcount_i
                lproposal_ratio = lhapcount_i - lhapcount
                log_accept[i] = ((llk_ratio + lprior_ratio) * temp + lproposal_ratio)                

    # calculate conditional probs of acceptance for all transitions
    conditionals = util.log_likelihoods_as_conditionals(log_accept)

    # random choice using probabilities
    choice = util.random_choice(conditionals)

    # update state
    genotype[h, j] = choice
    
    # return final log liklihood
    return llks[choice]

@numba.njit
def genotype_compound_step(genotype, reads, llk, mask=None, temp=1):
    """Mutation compound Gibbs sampler step for all base positions 
    of all haplotypes in a genotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as 
        simple integers from 0 to n_nucl.
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    llk : float
        Log-likelihood of the initial haplotype state given the 
        observed reads.
    mask : ndarray, bool, shape (n_base, n_nucl)
        Optionally indicate some alleles to skip e.g. alleles which 
        are known to have 0 probability.
    temp : float
        An inverse temperature in the interval 0, 1 to adjust
        the sampled distribution by.
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    ploidy, n_base = genotype.shape

    # matrix of haplotype-base combinations
    substeps = np.empty((ploidy * n_base, 2), dtype=np.int8) 

    for h in range(ploidy):
        for j in range(n_base):
            substep = (h * n_base) + j
            substeps[substep, 0] = h
            substeps[substep, 1] = j

    # random order
    np.random.shuffle(substeps)

    for i in range(ploidy * n_base):
        h, j = substeps[i]
        sub_mask = None if mask is None else mask[j]
        llk = base_step(genotype, reads, llk, h, j, sub_mask, temp=temp)
    return llk
