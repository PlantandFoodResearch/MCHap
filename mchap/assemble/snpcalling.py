import numpy as np
from numba import njit
from mchap.assemble.likelihood import log_likelihood
from mchap.jitutils import (
    normalise_log_probs,
    comb_with_replacement,
    increment_genotype,
)
from mchap.calling.prior import log_genotype_prior as log_snp_prior

__all__ = ["snp_posterior"]


@njit(cache=True)
def snp_posterior(
    read_probs, n_alleles, ploidy, flat_prior, inbreeding=0, read_counts=None
):
    """Brute-force the posterior probability across all possible
    genotypes for a single SNP position.

    Parameters
    ----------
    read_probs : ndarray, float, shape (n_reads, n_positions, max_allele)
        Reads probabilities at the given SNP.
    n_alleles : int
        Number of possible alleles for this SNP.
    ploidy : int
        Ploidy of organism.
    flat_prior : bool
        If true the inbreeding argument is ignored and a
        flat prior is assumed across all genotypes
    inbreeding : float
        Expected inbreeding coefficient of organism.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.

    Returns
    -------
    genotypes : ndarray, int, shape (n_genotypes, ploidy)
        SNP genotypes.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probability of each genotype.

    """
    n_reads, max_allele = read_probs.shape
    if n_reads == 0:
        # handle no reads
        n_reads = 1
        read_probs = np.empty((n_reads, max_allele), dtype=np.float64)
        read_probs[:] = np.nan

    u_gens = comb_with_replacement(n_alleles, ploidy)
    genotype = np.zeros(ploidy, dtype=np.int8)
    genotypes = np.empty((u_gens, ploidy), dtype=np.int8)
    log_probabilities = np.empty(u_gens, dtype=float)
    log_probabilities[:] = -np.inf
    for i in range(u_gens):
        genotypes[i] = genotype
        if flat_prior:
            lprior = 0.0
        else:
            lprior = log_snp_prior(
                genotype, unique_haplotypes=n_alleles, inbreeding=inbreeding
            )
        llk = log_likelihood(
            np.expand_dims(read_probs, 1),
            np.expand_dims(genotype, -1),
            read_counts=read_counts,
        )
        log_probabilities[i] = lprior + llk
        increment_genotype(genotype)
    # normalise
    probabilities = normalise_log_probs(log_probabilities)
    return genotypes, probabilities
