#!/usr/bin/env python3

import numpy as np
from math import factorial


def count_possible_alleles(reads):
    """Counts the number of possible alleles at each position
    in a set of probabilistically encoded reads.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_positions, max_allele)
        A set of reads encoded as probabilistic row vectors.

    Returns
    -------
    n_alleles : ndarray, int, shape (n_positions, )
        Number of possible alleles at each position given the set of reads.

    Notes
    -----
    An allele is only possible if one or more reads has a non-zero
    probability of beain that allele.
    Hence if all reads are zero-padded at an allele then that
    allele is not possible given this set of reads.

    """
    # distributions are padded by 0 prob
    check_all = np.sum(~np.all(np.nan_to_num(reads) == 0, axis=-3), axis=-1)
    check_any = np.sum(~np.any(reads == 0, axis=-3), axis=-1)
    
    # all read distributions should be in agreement
    if np.array_equal(check_all, check_any):
        return check_all
    
    else:
        raise ValueError('Incongruent allele distributions')


def count_unique_alleles(locus):
    """Count the number of alleles at each position of a locus.

    Parameters
    ----------
    locus : Locus
        A Locus object with attribute `alleles`

    Returns
    -------
    u_alleles : ndarray, int, shape (n_positions, )
        Number of unique alleles at each position defined by the locus.

    """
    return np.fromiter(map(len, locus.alleles), dtype=int)


def count_unique_haplotypes(u_alleles):
    """Calculate the number of unique haplotypes that can exist at 
    a locus given the number of unique alleles at each variable 
    position within the haplotype interval.

    Parameters
    ----------
    u_alleles : ndarray, int, shape (n_positions, )
        Number of unique alleles at each position defined by the locus.

    Returns
    -------
    u_haplotypes : int
        Number of possible unique haplotypes.

    """
    return np.prod(u_alleles)


def count_unique_genotypes(u_haps, ploidy):
    """Calculates number of possible unique genotypes at a locus
    given the number of possible unique haplotypes at that locus
    and a ploidy.

    Parameters
    ----------
    u_haps : int
        Number of possible unique haplotypes.
    ploidy : int
        Number of haplotype copys in an individual.

    Returns
    -------
    u_genotypes : int
        Number of possible unique genotypes excluding
        equivilent permutations.

    """
    return factorial(u_haps + ploidy -1) // (factorial(ploidy) * factorial(u_haps-1))


def count_unique_genotype_permutations(u_haps, ploidy):
    """Calculates number of possible genotypes at a locus (including
    equivilent permutations) given the number of possible unique 
    haplotypes at that locus and a ploidy.

    Parameters
    ----------
    u_haps : int
        Number of possible unique haplotypes.
    ploidy : int
        Number of haplotype copys in an individual.

    Returns
    -------
    genotype_perms : int
        Number of possible genotypes including
        equivilent permutations.

    """
    return u_haps ** ploidy


def count_haplotype_universial_occurance(u_haps, ploidy):
    """Counts the number of occurances of a haplotype among all 
    possible unique genotypes at a locus.

    Parameters
    ----------
    u_haps : int
        Number of possible unique haplotypes.
    ploidy : int
        Number of haplotype copys in an individual.
    
    Returns
    -------
    occurance : int
        Number of of time a single haplotype will occur among
        all possible genotypes in cluding equivilent permutations
        of genotypes.

    """
    return factorial(u_haps + ploidy -1) // (factorial(ploidy-1) * factorial(u_haps))


def count_genotype_permutations(dosage):
    """Counts the total number of equivilent genotype permutations 
    for a single given genotype.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy)
        Array with dose of each haplotype within a genotype.

    Returns
    -------
    genotype_perms : int
        Number of equivilent permutations for a
        genotype of the given dosage.

    Notes
    -----
    The sum of elements of the dosage should be equal to
    the ploidy of the genotype in question.

    """
    ploidy = sum(dosage)
    numerator = factorial(ploidy)
    denominator = 1
    for i in range(len(dosage)):
        denominator *= factorial(dosage[i])
    return numerator // denominator
