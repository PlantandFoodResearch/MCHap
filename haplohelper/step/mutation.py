import numpy as np 
from numba import jit

from haplohelper.step import util
from haplohelper.likelihood import log_likelihood

@jit(nopython=True)
def base_step(genotype, reads, llk, h, j):
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
    _, _, n_nucl = reads.shape
    
    # cache of log likelihoods calculated with each allele
    llks = np.empty(n_nucl)
    conditionals = np.empty(n_nucl)
    
    # reuse likelihood for current state
    current_nucleotide = genotype[h, j]
    llks[current_nucleotide] = llk
    
    for i in range(n_nucl):
        if i == current_nucleotide:
            # no need to recalculate likelihood
            pass
        else:
            genotype[h, j] = i
            llks[i] = log_likelihood(reads, genotype)

    # calculated denominator in log space
    log_denominator = llks[0]
    for i in range(1, n_nucl):
        log_denominator = util.add_log_prob(log_denominator, llks[i])

    # calculate conditional probabilities
    for i in range(n_nucl):
        conditionals[i] = np.exp(llks[i] - log_denominator)

    # ensure conditional probabilities are normalised 
    conditionals /= np.sum(conditionals)

    # if a prior is used then it can be multiplied by probs here
    choice = util.random_choice(conditionals)

    # update state
    genotype[h, j] = choice
    
    # return final log liklihood
    return llks[choice]


@jit(nopython=True)
def haplotype_compound_step(genotype, reads, llk, h):
    """Mutation compound Gibbs sampler step for all base positions of the hth haplotype.

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
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    _, n_base, _ = reads.shape
    
    for j in np.random.permutation(np.arange(0, n_base)):
        llk = base_step(genotype, reads, llk, h, j)
    return llk


@jit(nopython=True)
def genotype_compound_step(genotype, reads, llk):
    """Mutation compound Gibbs sampler step for all base positions of all haplotypes in a genotype.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    llk : float
        Log-likelihood of the initial haplotype state given the observed reads.
    
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
        llk = base_step(genotype, reads, llk, h, j)
    return llk
