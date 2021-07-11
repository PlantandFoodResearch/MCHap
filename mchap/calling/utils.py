import numpy as np
from numba import njit

from mchap.assemble.util import genotype_alleles_as_index


@njit(cache=True)
def allelic_dosage(genotype_alleles):
    """Return the dosage of genotype alleles encoded as integers.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Genotype alleles encoded as integers

    Returns
    -------
    dosage : ndarray, int, shape (ploidy, )
        Allelic dosage of genotype where dosage corresponds
        to the first instance of each allele.
    """
    ploidy = len(genotype_alleles)
    dosage = np.zeros(ploidy, dtype=genotype_alleles.dtype)

    for i in range(ploidy):
        a = genotype_alleles[i]
        searching = True
        j = 0
        while searching:
            if a == genotype_alleles[j]:
                dosage[j] += 1
                searching = False
            else:
                j += 1
    return dosage


@njit(cache=True)
def count_allele(genotype_alleles, allele):
    """Count occurrence of an allele in a genotype.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Genotype alleles encoded as integers
    allele : int
        Allele to count.

    Returns:
    count : int
        Count of allele.
    """
    count = 0
    for i in range(len(genotype_alleles)):
        if genotype_alleles[i] == allele:
            count += 1
    return count


@njit(cache=True)
def posterior_as_array(observed_genotypes, observed_probabilities, unique_genotypes):
    """Convert observed genotypes and their probabilities to an array of
    probabilities over all possible genotypes.

    Parameters
    ----------
    observed_genotypes : ndarray, int, shape (n_observed, ploidy)
        Alleles of observed genotypes.
    observed_probabilities : ndarray, float, shape (n_observed, )
        Probabilities associated with observed genotypes.
    unique_genotypes : int
        Number of total possible genotypes.

    Returns
    -------
    probabilities : ndarray, float, shape (unique_genotypes, )
        Probability of each possible genotype.
    """
    n_observed, _ = observed_genotypes.shape
    probabilities = np.zeros(unique_genotypes, np.float64)
    for i in range(n_observed):
        genotype = observed_genotypes[i]
        prob = observed_probabilities[i]
        idx = genotype_alleles_as_index(genotype)
        probabilities[idx] = prob
    return probabilities
