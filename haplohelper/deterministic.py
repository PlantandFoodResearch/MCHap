#!/usr/bin/env python3

import numpy as np
from itertools import  combinations as _combinations

import biovector as bv

from biovector.util import integer_vectors_as_probabilistic as _as_probs

from haplohelper import util
from haplohelper import inheritence


def logp(reads, haplotypes):
    mtx = bv.stats.probabalistic.pairwise(
        bv.stats.probabalistic.ibs_prob,
        reads,
        haplotypes)
    return np.sum(np.log(np.mean(mtx, axis=-1)))


def nan_logp(reads, haplotypes):
    mtx = bv.stats.probabalistic.pairwise(
        bv.stats.probabalistic.nan_ibs_prob,
        reads,
        haplotypes)
    return np.sum(np.log(np.mean(mtx, axis=-1)))


class GreedyHaplotypeModel(object):
    pass


class GreedyHaplotypeAssembler(GreedyHaplotypeModel):

    def __init__(self, ploidy=None):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.result = None

    def fit(self, reads, direction='middleout'):
        assert reads.ndim == 3

        ploidy = self.ploidy
        n_base = reads.shape[1]
        n_nucl = reads.shape[2]

        haps = np.zeros((ploidy, n_base, n_nucl))
        haps[:] = 0.25

        nucleotides = _as_probs(np.identity(n_nucl))

        assert direction in {'middleout', 'forward', 'reverse'}
        if direction == 'middleout':
            base_order = list(util.middle_out(range(n_base)))
        elif direction == 'forward':
            base_order = list(range(n_base))
        else:  # direction == 'reverse'
            base_order = list(reversed(range(n_base)))

        for hap in range(ploidy):
            for base in base_order:
                llk_opts = np.zeros(n_nucl)
                for nucl in range(n_nucl):
                    haps[hap, base] = nucleotides[nucl]
                    llk_opts[nucl] = logp(reads, haps)

                choice = np.argmax(llk_opts)
                haps[hap, base] = nucleotides[choice]

        self.result = haps


class GreedyRecombiner(GreedyHaplotypeModel):

    def __init__(self,
                 ploidy=None,
                 reference_haplotypes=None):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.reference_haplotypes = reference_haplotypes
        self.result = None

    def fit(self, reads, direction='middleout'):
        assert reads.ndim == 3

        ploidy, n_base, n_nucl = self.reference_haplotypes.shape
        assert self.ploidy == ploidy

        assert direction in {'middleout', 'forward', 'reverse'}
        if direction == 'middleout':
            base_order = list(util.middle_out(range(n_base)))
        elif direction == 'forward':
            base_order = list(range(n_base))
        else:  # direction == 'reverse'
            base_order = list(reversed(range(n_base)))

        pairs = list(_combinations(list(range(ploidy)), 2))

        result = self.reference_haplotypes.copy()

        # for each position try all hap combinations and pick best
        for pos in base_order:
            for i, (x, y) in enumerate(pairs):
                haps = result.copy()

                current = logp(reads, result)

                x_content = haps[x][pos:].copy()
                y_content = haps[y][pos:].copy()
                haps[x][pos:] = y_content
                haps[y][pos:] = x_content

                score = logp(reads, haps)

                if score > current:
                    result = haps

        self.result = result


class GreedyDosageCaller(GreedyHaplotypeModel):

    def __init__(self,
                 ploidy=None,
                 reference_haplotypes=None):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.reference_haplotypes = reference_haplotypes
        self.result = None

    def fit(self, reads):
        assert reads.ndim == self.reference_haplotypes.ndim == 3

        ploidy = self.ploidy
        n_base = reads.shape[1]
        n_nucl = reads.shape[2]

        # unique haplotypes
        known_haps = bv.mset.unique(self.reference_haplotypes)
        n_known_haps = len(known_haps)

        # start with null haplotypes
        haps = np.zeros((ploidy, n_base, n_nucl))
        haps[:] = 1/n_nucl

        for hap in range(ploidy):
            llk_opts = np.zeros(n_known_haps)

            for idx in range(n_known_haps):
                haps[hap] = known_haps[idx]
                llk_opts[idx] = logp(reads, haps)

            choice = np.argmax(llk_opts)
            haps[hap] = known_haps[choice]

        self.result = haps


class GreedyChildDosageCaller(GreedyHaplotypeModel):

    def __init__(self,
                 ploidy=None,
                 maternal_haplotypes=None,
                 paternal_haplotypes=None):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        assert maternal_haplotypes.shape == paternal_haplotypes.shape
        self.ploidy = ploidy
        self.maternal_haplotypes = maternal_haplotypes
        self.paternal_haplotypes = paternal_haplotypes
        self.result = None

    def fit(self, reads):
        assert reads.ndim == self.maternal_haplotypes.ndim == 3

        ploidy = self.ploidy
        n_base = reads.shape[1]
        n_nucl = reads.shape[2]

        trio = inheritence.TrioChildInheritance(
            self.maternal_haplotypes,
            self.paternal_haplotypes
        )

        haps = np.zeros((ploidy, n_base, n_nucl))
        haps[:] = 1/n_nucl
        for hap in range(ploidy):
            available_haps = trio.available()
            n_available_haps = len(available_haps)
            if n_available_haps == 0:
                pass

            else:
                llk_opts = np.zeros(n_available_haps)

                for idx in range(n_available_haps):
                    haps[hap] = available_haps[idx]
                    llk_opts[idx] = logp(reads, haps)

                choice = np.argmax(llk_opts)
                haps[hap] = available_haps[choice]
                trio.take(available_haps[choice], check=False)
        self.result = haps
