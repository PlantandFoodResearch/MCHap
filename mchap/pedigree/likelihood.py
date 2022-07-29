from numba import njit

from mchap.assemble.likelihood import log_likelihood
from mchap.jitutils import genotype_alleles_as_index


@njit(cache=True)
def log_likelihood_alleles_cached(
    reads, read_counts, haplotypes, sample, genotype_alleles, cache=None
):
    """
    Cached log-likelihood function for pedigree-based calling of genotypes.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    sample : int
        Unique index of sample.
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    cache : dict
        Cache of log-likeihoods whith keys of
        (sample_index, genotype_index) mapped
        to float.

    Returns
    -------
    llk : float
        Log-likelihood.
    """
    genotype_index = genotype_alleles_as_index(genotype_alleles)
    if cache is None:
        llk = log_likelihood(
            reads=reads,
            genotype=haplotypes[genotype_alleles],
            read_counts=read_counts,
        )
    else:
        key = (sample, genotype_index)
        if key in cache:
            llk = cache[key]
        else:
            llk = log_likelihood(
                reads=reads,
                genotype=haplotypes[genotype_alleles],
                read_counts=read_counts,
            )
            cache[key] = llk
    return llk
