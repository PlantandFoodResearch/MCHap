#!/usr/bin/env python3

import numpy as np 
import numba
from scipy import stats as _stats
from dataclasses import dataclass

from haplohelper import mset
from haplohelper.encoding import allelic
from haplohelper.encoding import probabilistic
from haplohelper.assemble.mcmc.step import mutation, structural
from haplohelper.assemble.likelihood import log_likelihood
from haplohelper.assemble import util, complexity
from haplohelper.assemble.classes import Assembler, GenotypeTrace


@numba.njit
def _denovo_gibbs_sampler(
        genotype, 
        reads, 
        mask,
        ratio, 
        steps, 
        break_dist,
        allow_recombinations,
        allow_dosage_swaps,
        allow_deletions
    ):
    _, n_base = genotype.shape
    
    llk = log_likelihood(reads, genotype)
    genotype_trace = np.empty((steps,) + genotype.shape, np.int8)
    llk_trace = np.empty(steps, np.float64)
    for i in range(steps):

        if np.isnan(llk):
            raise ValueError('Encountered log likelihood of nan')
        
        # chance of mutation step
        choice = ratio > np.random.random()
        
        if choice:
            # mutation step
            llk = mutation.genotype_compound_step(genotype, reads, llk, mask=mask)
        
        else:
            # structural step
            
            # choose number of break points
            n_breaks = util.random_choice(break_dist)
            
            # break into intervals
            intervals = structural.random_breaks(n_breaks, n_base)
            
            # compound step
            llk = structural.compound_step(
                genotype=genotype, 
                reads=reads, 
                llk=llk, 
                intervals=intervals,
                allow_recombinations=allow_recombinations,
                allow_dosage_swaps=allow_dosage_swaps,
                allow_deletions=allow_deletions
            )
        
        genotype_trace[i] = genotype.copy() # TODO: is this copy needed?
        llk_trace[i] = llk
    return genotype_trace, llk_trace


def _point_beta_probabilities(n_base, a=1, b=1):
    """Return probabilies for selecting a recombination point
    following a beta distribution

    Parameters
    ----------
    n_base : int
        Number of base positions in this genotype.
    a : float
        Alpha parameter for beta distribution.
    b : float
        Beta parameter for beta distribution.

    Returns
    -------
    probs : array_like, int, shape (n_base - 1)
        Probabilities for recombination point.
    
    """
    dist = _stats.beta(a, b)
    points = np.arange(1, n_base + 1) / (n_base)
    probs = dist.cdf(points)
    probs[1:] = probs[1:] - probs[:-1]
    return probs


def _read_mean_dist(reads):
    # work around to avoid nan values caused by gaps
    dist = reads.copy()
    dist[np.isnan(dist)] = 1
    dist /= np.expand_dims(dist.sum(axis=-1), -1)
    dist = np.mean(dist, axis=-3)
    return dist


@dataclass
class DenovoMCMC(Assembler):

    ploidy: int
    steps: int = 1000
    ratio: int = 0.75
    alpha: float = 1.0
    beta: float = 3.0
    n_intervals: int = None
    allow_recombinations: bool = True
    allow_dosage_swaps: bool = True
    allow_deletions: bool = False

    def fit(self, reads, initial=None):

        _, n_base, _ = reads.shape

        if initial is None:
            dist = _read_mean_dist(reads)
            # for each haplotype, take a random sample of mean of reads
            genotype = np.empty((self.ploidy, n_base), dtype=np.int8)
            for i in range(self.ploidy):
                genotype[i] = probabilistic.sample_alleles(dist)

        else:
            # use the provided array
            assert initial.shape == (self.ploidy, n_base)
            genotype = initial.copy()


        # distribution to draw number of break points from
        if self.n_intervals is None:
            # random number of intervals based on beta distribution
            break_dist = _point_beta_probabilities(n_base, self.alpha, self.beta)
        else:
            # this is a hack to fix number of intervals to a constant
            break_dist = np.zeros(self.n_intervals, dtype=np.float64)
            break_dist[-1] = 1  # 100% probability of n_intervals -1 break points

        # Automatically mask out alleles for which all 
        # reads have a probability of 0.
        # A probability of 0 indicates that the allele is invalid.
        # Masking alleles stops that allele being assessed in the
        # mcmc and hence reduces paramater space and compute time.
        mask = np.all(reads == 0, axis=-3)

        # run the sampler
        genotypes, llks = _denovo_gibbs_sampler(
            genotype, 
            reads, 
            mask=mask,
            ratio=self.ratio, 
            steps=self.steps, 
            break_dist=break_dist,
            allow_recombinations=self.allow_recombinations,
            allow_dosage_swaps=self.allow_dosage_swaps,
            allow_deletions=self.allow_deletions
        )

        return GenotypeTrace(genotypes, llks)
