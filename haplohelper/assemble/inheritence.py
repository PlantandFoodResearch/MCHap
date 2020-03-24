#!/usr/bin/env python3

import numpy as np
from itertools import combinations as _combinations
from collections import Counter as _Counter

from haplohelper import mset
from haplohelper.encoding import allelic


class TrioChildInheritance(object):

    def __init__(self, maternal_haplotypes, paternal_haplotypes):

        self.mum = maternal_haplotypes.copy()
        self.dad = paternal_haplotypes.copy()

        assert self.mum.ndim == self.dad.ndim == 3
        assert self.mum.shape == self.dad.shape
        assert self.mum.dtype == self.dad.dtype

        self.ploidy, self.n_base, self.n_nucl = self.mum.shape
        self.dtype = self.mum.dtype

        self.reset()

    def reset(self):
        self.taken = np.zeros((0, self.n_base, self.n_nucl), self.dtype)
        self.mum_taken = self.taken.copy()
        self.dad_taken = self.taken.copy()

    def available(self):
        result = np.zeros((0, self.n_base, self.n_nucl), self.dtype)
        if len(self.taken) == self.ploidy:
            return result
        if len(self.mum_taken) < (self.ploidy / 2):
            result = mset.add(result, mset.subtract(self.mum, self.mum_taken))
        if len(self.dad_taken) < (self.ploidy / 2):
            result = mset.add(result, mset.subtract(self.dad, self.dad_taken))
        return mset.unique(result)

    @staticmethod
    def _mset_thin(array, max_count):
        """Replacement for the deprecated thin in biovector"""
        unique, counts = mset.unique_counts(array)
        counts[counts > max_count] = max_count
        return mset.repeat(unique, counts)

    def take(self, hap, check=True):

        if hap.ndim == 2:
            hap = np.expand_dims(hap, 0)

        if check:
            assert mset.contains(self.available(), hap)

        self.taken = mset.add(self.taken, hap)

        improving = True
        while improving:
            improving = False

            # taken
            un_assigned = mset.subtract(self.taken,
                                        mset.add(self.mum_taken,
                                                 self.dad_taken))

            if len(self.dad_taken) == self.ploidy / 2:
                mum_update = mset.intercept(self.mum, un_assigned)
            else:
                dad_slots = (self.ploidy / 2) - len(self.dad_taken)
                dad_heritable = self._mset_thin(
                    mset.subtract(self.dad, self.dad_taken),
                    dad_slots
                )
                mum_update = mset.intercept(self.mum,
                                            mset.subtract(un_assigned,
                                                          dad_heritable))

            if len(mum_update) > 0:
                self.mum_taken = mset.add(self.mum_taken, mum_update)
                improving = True

            un_assigned = mset.subtract(self.taken,
                                        mset.add(self.mum_taken,
                                                 self.dad_taken))

            if len(self.mum_taken) == self.ploidy / 2:
                dad_update = mset.intercept(self.dad, un_assigned)
            else:
                mum_slots = (self.ploidy / 2) - len(self.mum_taken)
                mum_heritable = self._mset_thin(
                    mset.subtract(self.mum, self.mum_taken),
                    mum_slots
                )
                dad_update = mset.intercept(self.dad,
                                            mset.subtract(un_assigned,
                                                          mum_heritable))

            if len(dad_update) > 0:
                self.dad_taken = mset.add(self.dad_taken, dad_update)
                improving = True


def gamete_probabilities(genotypes,
                         probabilities,
                         order=None):
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

