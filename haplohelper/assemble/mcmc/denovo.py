#!/usr/bin/env python3

import numpy as np 
import numba

from haplohelper import mset
from haplohelper.encoding import allelic
from haplohelper.encoding import probabilistic
from haplohelper.assemble.step import util, mutation, structural
from haplohelper.assemble.likelihood import log_likelihood
from haplohelper.assemble import complexity
from haplohelper.util import point_beta_probabilities


@numba.njit
def _denovo_gibbs_sampler(
        genotype, 
        reads, 
        n_alleles,
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
            llk = mutation.genotype_compound_step(genotype, reads, llk, n_alleles)
        
        else:
            # structural step
            
            # choose number of break points
            n_breaks = util.random_choice(break_dist)
            
            # break into intervals
            intervals = util.random_breaks(n_breaks, n_base)
            
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


class DeNovoGibbsAssembler(object):
    
    def __init__(
            self, 
            ploidy, 
            steps=1000, 
            initial=None, 
            ratio=0.75, 
            alpha=1, 
            beta=3,
            n_intervals=None,
            allow_recombinations=True,
            allow_dosage_swaps=True,
            allow_deletions=False
        ):
        
        if n_intervals:
            # the following are not used
            alpha = None
            beta = None
            
        if ratio == 1:
            # mutation only model
            alpha = None
            beta = None
            n_intervals = None

        self.ploidy = ploidy
        self.n_base = None
        self.n_nucl = None
        self.steps = steps
        self.initial = initial
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta
        self.n_intervals = n_intervals
        self.allow_recombinations = allow_recombinations
        self.allow_dosage_swaps = allow_dosage_swaps
        self.allow_deletions = allow_deletions
        
        self.genotype_trace = None
        self.llk_trace = None
        
        
    def fit(self, reads):
        
        _, n_base, n_nucl = reads.shape
        
        self.n_base = n_base
        self.n_nucl = n_nucl

        n_alleles = complexity.count_unique_alleles(reads)
        
        # initial state
        if self.initial is None:
            # random sample of mean of reads
            genotype = np.empty((self.ploidy, n_base), dtype=np.int8)
            for i in range(self.ploidy):
                genotype[i] = probabilistic.sample_alleles(reads.mean(axis=-3))
        else:
            genotype = self.initial.copy()
            
        # distribution to draw number of break points from
        if self.n_intervals is None:
            # random number of intervals based on beta distribution
            break_dist = point_beta_probabilities(n_base, self.alpha, self.beta)
        else:
            # this is a hack to fix number of intervals
            break_dist = np.zeros(self.n_intervals, dtype=np.float64)
            break_dist[-1] = 1  # 100% probability of n_intervals -1 break points
        
        self.genotype_trace, self.llk_trace = _denovo_gibbs_sampler(
            genotype, 
            reads, 
            n_alleles,
            ratio=self.ratio, 
            steps=self.steps, 
            break_dist=break_dist,
            allow_recombinations=self.allow_recombinations,
            allow_dosage_swaps=self.allow_dosage_swaps,
            allow_deletions=self.allow_deletions
        )
        
    def sorted_trace(self):
        trace = self.genotype_trace.copy()
        for i in range(len(trace)):
            trace[i] = allelic.sort(trace[i])
        return trace
    
    def posterior(self, burn=0, probabilities=True, sort=True):
        
        # burn sorted trace
        trace = self.sorted_trace()[burn:]
        
        # unique genotypes and their counts
        genotypes, counts = mset.unique_counts(trace)
        
        if probabilities:
            probs = counts / np.sum(counts)
        else:
            probs = counts
        
        if sort:
            idx = np.flip(np.argsort(probs))
            return genotypes[idx], probs[idx]
        else:
            return genotypes, probs
