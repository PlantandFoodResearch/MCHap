import numpy as np
from numba import njit

from mchap.combinatorics import count_unique_genotypes
from mchap.assemble.likelihood import log_likelihood, log_genotype_prior
from mchap.assemble.util import get_dosage, normalise_log_probs


__all__ = [
    "genotype_likelihoods",
    "genotype_posteriors",
]


@njit
def increment_genotype(genotype):
    """Increment a genotype of allele numbers to the next genotype
    in VCF sort order.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy,)
        Array of allele numbers in the genotype.

    Notes
    -----
    Mutates genotype array in place.
    """
    ploidy = len(genotype)
    if ploidy == 1:
        # haploid case
        genotype[0] += 1
        return
    previous = genotype[0]
    for i in range(1, ploidy):
        allele = genotype[i]
        if allele == previous:
            pass
        elif allele > previous:
            i -= 1
            genotype[i] += 1
            genotype[0:i] = 0
            return
        else:
            raise ValueError("genotype alleles are not in ascending order")
    # all alleles are equal
    genotype[-1] += 1
    genotype[0:-1] = 0


@njit
def _genotype_likelihoods(reads, ploidy, haplotypes, n_genotypes, read_counts=None):
    likelihoods = np.full(n_genotypes, np.nan, np.float32)
    genotype = np.zeros(ploidy, np.int64)
    for i in range(0, n_genotypes):
        likelihoods[i] = log_likelihood(
            reads=reads,
            genotype=haplotypes[genotype],
            read_counts=read_counts,
        )
        increment_genotype(genotype)
    return likelihoods


def genotype_likelihoods(reads, ploidy, haplotypes, read_counts=None):
    """Calculate the log likelihood of every possible genotype
    for a given set of reads, ploidy, and possible haplotypes.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        A set of probabalistically encoded reads.
    ploidy : int
        Ploidy of organism.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes in VCF allele order.

    Returns
    -------
    log_likelihoods : ndarray, float, shape (n_genotypes, )
        VCF ordered genotype log likelihoods.
    """
    n_haplotypes = len(haplotypes)
    n_genotypes = count_unique_genotypes(n_haplotypes, ploidy)
    return _genotype_likelihoods(
        reads=reads,
        ploidy=ploidy,
        haplotypes=haplotypes,
        n_genotypes=n_genotypes,
        read_counts=read_counts,
    )


@njit
def genotype_posteriors(log_likelihoods, ploidy, n_alleles, inbreeding=0):
    """Calculate posterior probability of every possible genotype
    for a given set of likelihoods, ploidy, and number of alleles.

    Parameters
    ----------
    log_likelihoods : ndarray, float, shape(n_genotypes, )
        VCF ordered log (natural) likelihood of each possible genotype.
    ploidy : int
        Ploidy of organism.
    n_alleles : int
        Total number of possible (haplotype) alleles at this locus.
    inbreeding : float
        Inbreeding coefficient of organism used when calculating prior
        probabilities.

    Returns
    -------
    posteriors : ndarray, float, shape(n_genotypes, )
        VCF ordered posterior probability of each possible genotype.
    """
    n_genotypes = len(log_likelihoods)
    posteriors = np.zeros(n_genotypes, dtype=log_likelihoods.dtype)
    genotype = np.zeros(ploidy, np.int64)
    dosage = np.zeros(ploidy, np.int8)
    for i in range(n_genotypes):
        get_dosage(dosage, genotype.reshape(ploidy, 1))
        llk = log_likelihoods[i]
        lpr = log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
        posteriors[i] = llk + lpr
        increment_genotype(genotype)
    return normalise_log_probs(posteriors)
