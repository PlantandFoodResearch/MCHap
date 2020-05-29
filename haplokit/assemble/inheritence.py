#!/usr/bin/env python3

import numpy as np
from itertools import combinations as _combinations
from collections import Counter as _Counter

from haplokit import mset
from haplokit.encoding import allelic


def gamete_probabilities(genotypes,
                         probabilities,
                         order=None):
    """Calculate unique gametes and their associated probabilities
    given a distribution of genotypes and probabilities for an
    individual.

    Parameters
    ----------
    genotypes : ndarray, int, shape (n_genotypes, ploidy, n_positions)
        The possible genotypes of an individual of known ploidy at a 
        locus covering n_positions variable positions.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probabilities (summing to 1) of each genotype.
    order : str, optional
        Specify if the gametes should be returned in 'ascending' or 
        'descending' order of probabilities.

    Returns
    -------
    gametes : ndarray, int, shape (n_gametes, ploidy//2, n_positions)
        The possible gametes produced by the individual.
    gamete_probabilities : ndarray, float, shape (n_gametes, )
        Probabilities (summing to 1) of each gamete.

    """

    assert order in {None, 'ascending', 'descending'}

    n_gens, ploidy, n_base = genotypes.shape

    # convert haplotypes to strings for faster hashing/comparisons
    string_to_hap = {}
    genotype_strings = np.empty(n_gens * ploidy, dtype='<O')
    for i, hap in enumerate(genotypes.reshape(n_gens * ploidy, n_base)):
        string = hap.tostring()
        string_to_hap[string] = hap
        genotype_strings[i] = string
    genotype_strings = np.sort(genotype_strings.reshape(n_gens, ploidy), axis=-1)

    # calculate gamete probs
    gamete_probs = {}
    for genotype_string, set_prob in zip(genotype_strings, probabilities):
        gametes = list(_combinations(genotype_string, ploidy // 2))
        n_gametes = len(gametes)
        for gamete, count in _Counter(gametes).items():
            prob = set_prob * (count / n_gametes)
            if gamete in gamete_probs:
                gamete_probs[gamete] += prob
            else:
                gamete_probs[gamete] = prob

    # convert from hash map of string: prob back to arrays
    n_gametes = len(gamete_probs)
    new = np.empty((n_gametes, ploidy // 2, n_base), dtype=np.int8)
    new_probs = np.empty(n_gametes, dtype=np.float)
    for i, (strings, prob) in enumerate(gamete_probs.items()):
        new_probs[i] = prob
        for j, string in enumerate(strings):
            new[i, j] = string_to_hap[string]

    if order:
        idx = np.argsort(new_probs)
        if order == 'descending':
            idx = np.flip(idx, axis=0)
        return new[idx], new_probs[idx]
    else:
        return new, new_probs


def cross_probabilities(maternal_gametes,
                        maternal_probabilities,
                        paternal_gametes,
                        paternal_probabilities,
                        order=None):
    """Calculate unique genotypes and their associated probabilities
    given distributions of maternal and paternal gametes.

    Parameters
    ----------
    maternal_gametes : ndarray, int, shape (n_gametes, ploidy//2, n_positions)
        The possible maternal gametes contributing to an individual 
        at a locus covering n_positions variable positions.
    maternal_probabilities : ndarray, float, shape (n_gametes, )
        Probabilities (summing to 1) of each maternal gamete.
    paternal_gametes : ndarray, int, shape (n_gametes, ploidy//2, n_positions)
        The possible paternal gametes contributing to an individual 
        at a locus covering n_positions variable positions.
    paternal_probabilities : ndarray, float, shape (n_gametes, )
        Probabilities (summing to 1) of each paternal gamete.
    order : str, optional
        Specify if the genotypes should be returned in 'ascending' or 
        'descending' order of probabilities.

    Returns
    -------
    genotypes : ndarray, int, shape (n_genotypes, ploidy, n_positions)
        The possible genotypes of an individual of known ploidy at a 
        locus covering n_positions variable positions.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probabilities (summing to 1) of each genotype.

    """

    assert order in {None, 'ascending', 'descending'}

    # get dimensions
    half_ploidy, n_base = maternal_gametes.shape[-2:]
    ploidy = half_ploidy * 2

    # compute genotypes and probabilities
    genotype_probs = {}
    string_to_genotype = {}
    for m_gamete, m_prob in zip(maternal_gametes,
                                maternal_probabilities):
        for p_gamete, p_prob in zip(paternal_gametes,
                                    paternal_probabilities):
            genotype = np.empty((ploidy, n_base), dtype=np.int8)
            idx = 0
            for gamete in [m_gamete, p_gamete]:
                for hap in gamete:
                    genotype[idx] = hap
                    idx += 1

            genotype = allelic.sort(genotype)
            string = genotype.tostring()
            if string not in string_to_genotype:
                string_to_genotype[string] = genotype

            prob =  m_prob * p_prob
            if string in genotype_probs:
                genotype_probs[string] += prob
            else:
                genotype_probs[string] = prob

    # convert genotypes bac to arrays
    n_genotypes = len(genotype_probs)
    new = np.empty((n_genotypes, ploidy, n_base), dtype=np.int8)
    new_probs = np.empty(n_genotypes, dtype=np.float)

    for i, (string, prob) in enumerate(genotype_probs.items()):
        new_probs[i] = prob
        new[i] = string_to_genotype[string]

    if order:
        idx = np.argsort(new_probs)
        if order == 'descending':
            idx = np.flip(idx, axis=0)
        return new[idx], new_probs[idx]
    else:
        return new, new_probs

