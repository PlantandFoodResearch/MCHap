import numpy as np
from numba import njit

from mchap.assemble.likelihood import log_likelihood
from mchap.assemble.util import genotype_alleles_as_index


@njit(cache=True)
def log_likelihood_alleles(reads, read_counts, haplotypes, genotype_alleles):
    """Log-likelihood function for genotype alleles indexing a
    set of known haplotypes.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.

    Returns
    -------
    llk : float
        Log-likelihood.
    """
    return log_likelihood(
        reads=reads,
        genotype=haplotypes[genotype_alleles],
        read_counts=read_counts,
    )


@njit(cache=True)
def log_likelihood_alleles_cached(
    reads, read_counts, haplotypes, genotype_alleles, cache
):
    """
    Cached log-likelihood function for genotype alleles indexing a
    set of known haplotypes.
    ----------
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    cache : dict
        Cache of log-likelihoods mapping genotype index (int) to llk (float).
    Returns
    -------
    llk : float
        Log-likelihood.
    """
    key = genotype_alleles_as_index(np.sort(genotype_alleles))
    if key in cache:
        llk = cache[key]
    else:
        llk = log_likelihood_alleles(
            reads=reads,
            read_counts=read_counts,
            haplotypes=haplotypes,
            genotype_alleles=genotype_alleles,
        )
        cache[key] = llk
    return llk
