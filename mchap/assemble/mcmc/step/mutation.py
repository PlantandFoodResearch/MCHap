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

    # cache of log likelihoods calculated with each allele
    llks = np.empty(n_alleles)
    
    # reuse likelihood for current state
    current_nucleotide = genotype[h, j]
    llks[current_nucleotide] = llk
    
    for i in range(n_alleles):
        if (mask is not None) and mask[i]:
            # skip masked allele
            llks[i] = - np.inf
        else:
            if i == current_nucleotide:
                # no need to recalculate likelihood
                pass
            else:
                genotype[h, j] = i
                llks[i] = log_likelihood(reads, genotype)

    # calculate conditional probabilities
    conditionals = util.log_likelihoods_as_conditionals(llks)

    # update probabilities with temperature
    # TODO: could apply temp to llks to avoid second normalisation
    if temp < 1:
        conditionals **= temp
        conditionals /= conditionals.sum()

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
