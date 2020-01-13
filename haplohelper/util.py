#!/usr/bin/env python3

import numpy as np
from itertools import islice as _islice
from itertools import zip_longest as _zip_longest
from math import factorial as _factorial

import biovector


def prob_of_qual(qual):
    return 1 - (10 ** (qual / -10))


def qual_of_prob(prob):
    return int(-10 * np.log10((1 - prob)))


def flatten(item, container):
    if isinstance(item, container):
        for element in item:
            yield from flatten(element, container)
    else:
        yield item


def merge(*args):
    for tup in _zip_longest(*args):
        for val in tup:
            if val is not None:
                yield val


def middle_out(sequence):
    gen = (i for i in sequence)
    first_half = list(_islice(gen, len(sequence) // 2))
    second_half = list(gen)
    return merge(second_half, reversed(first_half))


def suggest_alphabet(vector_size):
    if vector_size == 2:
        return biovector.alphabet.biallelic
    elif vector_size == 3:
        return biovector.alphabet.triallelic
    elif vector_size == 4:
        return biovector.alphabet.quadraallelic
    else:
        raise ValueError(
            'No suitable alphabet for vector size {}'.format(vector_size))


def count_unique_haplotypes(n_base, n_nucl):
    return n_base ** n_nucl


def count_unique_genotypes(u_haps, ploidy):
    return _factorial(u_haps + ploidy -1) // (_factorial(ploidy) * _factorial(u_haps-1))


def count_haplotype_universial_occurance(u_haps, ploidy):
    """Counts the number of ocurances of a haplotype among all possible unique genotypes.
    """
    return _factorial(u_haps + ploidy -1) // (_factorial(ploidy-1) * _factorial(u_haps))


def count_genotype_perterbations(dosage):
    """Counts the total number of equivilent genotype perterbation based on the dosage.

    A genotype is an unsorted set of haplotypes hence the genotype `{A, B}` is equivielnt
    to the genotype `{B, A}`.
    A fully homozygous genotype e.g. `{A, A}` has only one possible perterbation.
    """
    ploidy = np.sum(dosage)
    numerator = _factorial(ploidy)
    denominator = np.prod([_factorial(d) for d in dosage])
    return numerator // denominator
