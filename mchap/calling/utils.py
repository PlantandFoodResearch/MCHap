import numpy as np
from numba import njit


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
