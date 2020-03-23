#!/usr/bin/env python3
from math import factorial

def count_unique_haplotypes(n_base, n_nucl):
    """Calculates number of unique haplotypes based on constraints.
    """
    return n_nucl ** n_base


def count_unique_genotypes(u_haps, ploidy):
    """Calculates number of unique genotypes based on constraints,
    not including equivilent permutations.
    """
    return factorial(u_haps + ploidy -1) // (factorial(ploidy) * factorial(u_haps-1))


def count_unique_genotype_permutations(ploidy, n_base, n_nucl):
    """Calculates number of genotypes based on constraints,
    including equivilent permutations.
    """
    return (n_nucl ** n_base) ** ploidy


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
