#!/usr/bin/env python3

import numpy as np
from math import factorial
from scipy.special import comb

__all__ = [
    "count_unique_haplotypes",
    "count_unique_genotypes",
    "count_unique_genotypes",
    "count_unique_genotype_permutations",
    "count_genotype_permutations",
]


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
    return int(comb(u_haps, ploidy, repetition=True))


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
    return u_haps**ploidy


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
    return factorial(u_haps + ploidy - 1) // (factorial(ploidy - 1) * factorial(u_haps))


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
