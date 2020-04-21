#!/usr/bin/env python3

import numpy as np 
import numba
from scipy import stats as _stats

from haplohelper import mset
from haplohelper.encoding import allelic
from haplohelper.encoding import probabilistic
from haplohelper.assemble.mcmc.step import mutation, structural
from haplohelper.assemble.likelihood import log_likelihood
from haplohelper.assemble import util, complexity


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


def denovo_mcmc(
        reads,
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
    
    _, n_base, _ = reads.shape

    # Automatically mask out alleles for which all 
    # reads have a probability of 0.
    # A probability of 0 indicates that the allele is invalid.
    # Masking alleles stops that allele being assessed in the
    # mcmc and hence reduces paramater space and compute time.
    mask = np.all(reads == 0, axis=-3)
    
    # initial genotype state
    if initial is None:
        # random sample of mean of reads
        genotype = np.empty((ploidy, n_base), dtype=np.int8)
        dist = np.nanmean(reads, axis=-3)
        for i in range(ploidy):
            genotype[i] = probabilistic.sample_alleles(dist)
        # if there is a compleate absence of observations for a position it will be a gap
        # so replace gaps with reference allele to be sure
        genotype[genotype < 0] = 0

    else:
        # use the provided array
        assert initial.shape == (ploidy, n_base)
        genotype = initial.copy()
        
    # distribution to draw number of break points from
    if n_intervals is None:
        # random number of intervals based on beta distribution
        break_dist = _point_beta_probabilities(n_base, alpha, beta)
    else:
        # this is a hack to fix number of intervals to a constant
        break_dist = np.zeros(n_intervals, dtype=np.float64)
        break_dist[-1] = 1  # 100% probability of n_intervals -1 break points
    
    # run the sampler
    genotype_trace, llk_trace = _denovo_gibbs_sampler(
        genotype, 
        reads, 
        mask=mask,
        ratio=ratio, 
        steps=steps, 
        break_dist=break_dist,
        allow_recombinations=allow_recombinations,
        allow_dosage_swaps=allow_dosage_swaps,
        allow_deletions=allow_deletions
    )

    # ensure haplotypes within each genotype are in consistant order
    # this is important for calculating posteriors etc
    for i in range(len(genotype_trace)):
        genotype_trace[i] = allelic.sort(genotype_trace[i])

    return genotype_trace, llk_trace


def posterior_dist(trace, burn=0, probabilities=True, sort=True):
    # TODO: should this be generic for traces from multiple sorces?

    # burn sorted trace
    trace = trace[burn:]
    
    # unique states and their counts
    states, counts = mset.unique_counts(trace)
    
    if probabilities:
        probs = counts / np.sum(counts)
    else:
        probs = counts
    
    if sort:
        idx = np.flip(np.argsort(probs))
        return states[idx], probs[idx]
    else:
        return states, probs


def posterior_max(trace, burn=0):
    states, probs = posterior_dist(trace, burn=burn)
    return states[0], probs[0]


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
        
        self.genotype_trace, self.llk_trace = denovo_mcmc(
                reads,
                ploidy=self.ploidy,
                steps=self.steps,
                initial=self.initial,
                ratio=self.ratio,
                alpha=self.alpha, 
                beta=self.beta,
                n_intervals=self.n_intervals,
                allow_recombinations=self.allow_recombinations,
                allow_dosage_swaps=self.allow_dosage_swaps,
                allow_deletions=self.allow_deletions
            )

    def posterior(self, burn=0, probabilities=True, sort=True):
        return posterior_dist(
            self.genotype_trace, 
            burn=burn, 
            probabilities=probabilities, 
            sort=sort
        )

    def best(self, burn=0):
        return posterior_max(self.genotype_trace, burn=burn)
