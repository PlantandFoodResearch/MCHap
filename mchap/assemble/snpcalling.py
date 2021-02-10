import numpy as np
from itertools import combinations_with_replacement
from mchap.combinatorics import count_unique_genotypes
from mchap.assemble.likelihood import log_likelihood, log_genotype_prior

__all__ = ["snp_posterior"]


def snp_posterior(reads, position, n_alleles, ploidy, inbreeding=0):
    """Brute-force the posterior probability across all possible
    genotypes for a single SNP position.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_positions, max_allele)
        Reads encoded as probability distributions.
    position : int
        Position of target SNP within reads.
    n_alleles : int
        Number of possible alleles for this SNP.
    ploidy : int
        Ploidy of organism.
    inbreeding : float
        Expected inbreeding coefficient of organism.

    Returns
    -------
    genotypes : ndarray, int, shape (n_genotypes, ploidy)
        SNP genotypes.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probability of each genotype.

    """
    n_reads, n_pos, max_allele = reads.shape
    if n_reads == 0:
        # handle no reads
        n_reads = 1
        reads = np.empty((n_reads, n_pos, max_allele), dtype=float)
        reads[:] = np.nan

    u_gens = count_unique_genotypes(n_alleles, ploidy)
    genotypes = np.zeros((u_gens, ploidy), dtype=np.int8) - 1
    probabilities = np.zeros(u_gens, dtype=float)

    alleles = np.arange(n_alleles)
    for j, genotype in enumerate(combinations_with_replacement(alleles, ploidy)):
        genotype = np.array(genotype)
        genotypes[j] = genotype
        _, dosage = np.unique(genotype, return_counts=True)
        lprior = log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
        # treat as haplotypes with single position
        llk = log_likelihood(reads[:, position : position + 1, :], genotype[..., None])

        probabilities[j] = np.exp(lprior + llk)

    # normalise
    probabilities /= np.nansum(probabilities)
    return genotypes, probabilities
