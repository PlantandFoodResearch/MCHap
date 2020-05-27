#!/usr/bin/env python3

import numpy as np
from math import factorial


def count_possible_alleles(reads):
    # distributions are padded by 0 prob
    check_all = np.sum(~np.all(np.nan_to_num(reads) == 0, axis=-3), axis=-1)
    check_any = np.sum(~np.any(reads == 0, axis=-3), axis=-1)
    
    # all read distributions should be in agreement
    if np.array_equal(check_all, check_any):
        return check_all
    
    else:
        raise ValueError('Incongruent allele distributions')


def count_unique_alleles(locus):
    return np.fromiter(map(len, locus.alleles), dtype=np.int)


def count_unique_haplotypes(u_alleles):
    """Calculates number of unique haplotypes based on constraints.
    """
    return np.prod(u_alleles)


def count_unique_genotypes(u_haps, ploidy):
    """Calculates number of unique genotypes based on constraints,
    not including equivilent permutations.
    """
    return factorial(u_haps + ploidy -1) // (factorial(ploidy) * factorial(u_haps-1))


def count_unique_genotype_permutations(u_haps, ploidy):
    """Calculates number of genotypes based on constraints,
    including equivilent permutations.
    """
    return u_haps ** ploidy


def count_haplotype_universial_occurance(u_haps, ploidy):
    """Counts the number of ocurances of a haplotype among all possible unique genotypes.
    """
    return factorial(u_haps + ploidy -1) // (factorial(ploidy-1) * factorial(u_haps))


def count_genotype_permutations(dosage):
    """Counts the total number of equivilent genotype permutations for a given genotype.

    A genotype is an unsorted set of haplotypes hence the genotype `{A, B}` is equivielnt
    to the genotype `{B, A}`.
    A fully homozygous genotype e.g. `{A, A}` has only one possible permutations.
    """
    ploidy = sum(dosage)
    numerator = factorial(ploidy)
    denominator = 1
    for i in range(len(dosage)):
        denominator *= factorial(dosage[i])
    return numerator // denominator
