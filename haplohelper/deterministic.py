#!/usr/bin/env python3

import numpy as np
import biovector as bv

from haplohelper import util
from haplohelper import inheritence


def llk(reads, haplotypes):
    return np.sum(np.log(np.mean(bv.stats.pairwise(bv.stats.pid,
                                                   reads,
                                                   haplotypes), axis=-1)))


class GreedyHaplotypeAssembler(object):

    def __init__(self, ploidy=None):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.result = None

    def haplotypes(self):
        return self.result.copy().astype(np.int8)

    def fit(self, reads, direction='middleout'):
        assert reads.ndim == 3

        ploidy = self.ploidy
        n_base = reads.shape[1]
        n_nucl = reads.shape[2]

        haps = np.zeros((ploidy, n_base, n_nucl))
        haps[:] = 0.25

        nucleotides = bv.as_probabilities(np.identity(n_nucl))

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
                    llk_opts[nucl] = llk(reads, haps)

                choice = np.argmax(llk_opts)
                haps[hap, base] = nucleotides[choice]

        self.result = haps


class GreedyDosageCaller(object):

    def __init__(self,
                 ploidy=None,
                 reference_haplotypes=None):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.reference_haplotypes = reference_haplotypes
        self.result = None

    def haplotypes(self):
        return self.result.copy().astype(np.int8)

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
                llk_opts[idx] = llk(reads, haps)

            choice = np.argmax(llk_opts)
            haps[hap] = known_haps[choice]

        self.result = haps


class GreedyChildDosageCaller(object):

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

    def haplotypes(self):
        return self.result.copy().astype(np.int8)

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
                    llk_opts[idx] = llk(reads, haps)

                choice = np.argmax(llk_opts)
                haps[hap] = available_haps[choice]
                trio.take(available_haps[choice], check=False)
        self.result = haps
