#!/usr/bin/env python3

import numpy as np 
import numba

from haplohelper.assemble.step import util
from haplohelper.assemble.likelihood import log_likelihood

@numba.njit
def base_step(genotype, reads, llk, h, j, n_alleles=None):
    """Mutation Gibbs sampler step for the jth base position of the hth haplotype.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    llk : float
        Log-likelihood of the initial haplotype state given the observed reads.
    h : int
        Index of the haplotype to be muteated.
    j : int
        Index of the base position to be mutated
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    if n_alleles is None:
        n_alleles = reads.shape[-1]

    # cache of log likelihoods calculated with each allele
    llks = np.empty(n_alleles)
    conditionals = np.empty(n_alleles)
    
    # reuse likelihood for current state
    current_nucleotide = genotype[h, j]
    llks[current_nucleotide] = llk
    
    for i in range(n_alleles):
        if i == current_nucleotide:
            # no need to recalculate likelihood
            pass
        else:
            genotype[h, j] = i
            llks[i] = log_likelihood(reads, genotype)

    # calculate conditional probabilities
    conditionals = util.log_likelihoods_as_conditionals(llks)

    # if a prior is used then it can be multiplied by probs here
    choice = util.random_choice(conditionals)

    # update state
    genotype[h, j] = choice
    
    # return final log liklihood
    return llks[choice]

@numba.njit
def genotype_compound_step(genotype, reads, llk, n_alleles=None):
    """Mutation compound Gibbs sampler step for all base positions of all haplotypes in a genotype.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    llk : float
        Log-likelihood of the initial haplotype state given the observed reads.
    n_alleles : array_like, int, shape (n_base, )
        Number of alleles to sample from at each step. Vector size used by default
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    ploidy, n_base = genotype.shape

    if n_alleles is None:
        n_alleles = np.repeat(reads.shape[-1], n_base)

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
        llk = base_step(genotype, reads, llk, h, j, n_alleles[j])
    return llk
