#!/usr/bin/env python3

import numpy as np 
import numba
from scipy import stats as _stats
from itertools import combinations_with_replacement as _combinations_with_replacement
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


def _homozygosity_probabilities(reads, ploidy):
    """Returns posterior probabilities of each base position being homozygous for each alllele.
    THe posterior probability is calculated for each base position independently of other base positions.
    """
    # probability of homozygousity for each allele at each base position
    probs = np.zeros(reads.shape[-2:], dtype=reads.dtype)
    
    # mask out zeros used to pad ragged arrays
    mask = np.all(reads == 0, axis=-3)
    
    # iterate through each base position
    for i in range(len(probs)):
        # number of valid alleles at this base position
        n_alleles = np.sum(~mask[i]).astype(np.int)
        
        # vector of unique alleles 
        alleles = np.arange(n_alleles, dtype=np.int8)
        
        # array of every possible genotype (at this single base position)
        genotypes = list(_combinations_with_replacement(alleles, ploidy))
        genotypes = np.expand_dims(np.array(genotypes, dtype=np.int8), -1)
        
        # read probabilities for this base position
        sub_reads = reads[:, i:i+1, :]
        
        # array to store llks
        llks = np.empty(len(genotypes), dtype = np.float)
        
        # array to store dosage
        dosage = np.ones(ploidy, dtype=np.int8)
        
        # keep indices of homozygous genotypes
        homozygous_genotypes = []
        
        for j in range(len(llks)):
            
            # calculate llk
            llks[j] = log_likelihood(sub_reads, genotypes[j])
            
            # adjust llk for possible permutations of this genotype based on it's dosage
            util.get_dosage(dosage, genotypes[j])
            perms = complexity.count_genotype_permutations(dosage)
            llks[j] += np.log(perms)
            
            # homozygous genotypes only have a single permutation
            if perms == 1:
                homozygous_genotypes.append(j)
        
        # insert probabilities for homozygous genotypes into the returned array
        probs[i, 0:n_alleles] = util.log_likelihoods_as_conditionals(llks)[homozygous_genotypes]
        
    return probs



@dataclass
class DenovoMCMC(Assembler):

    ploidy: int
    steps: int = 1000
    ratio: int = 0.75
    alpha: float = 1.0
    beta: float = 3.0
    n_intervals: int = None
    fix_homozygous: float = 0.999
    allow_recombinations: bool = True
    allow_dosage_swaps: bool = True
    allow_deletions: bool = True

    def fit(self, reads, initial=None):

        # identify base positions that are overwhelmingly likely
        # to be homozygous
        # these can be 'fixed' to reduce computational complexity
        hom_probs = _homozygosity_probabilities(reads, self.ploidy)
        fixed = hom_probs >= self.fix_homozygous
        homozygous = np.any(fixed, axis=-1)
        heterozygous = ~homozygous

        # subset of read positions which have not been identified 
        # as homozygous
        reads_het = reads[:, heterozygous]

        # counts of total number of bases and number of het bases
        _, n_base, _ = reads.shape
        _, n_het_base, _ = reads_het.shape

        # if all bases are fixed homozygous we don't need to sample anything
        if n_het_base == 0:
            # create a haplotype of the fixed alleles
            idx, vals = np.where(fixed)
            haplotype = np.zeros(n_base, dtype=np.int8)
            haplotype[idx] = vals
            # tile for each haplotype in each "step"
            genotypes = np.tile(haplotype, (self.steps, self.ploidy, 1))
            # set likelihoods to nan
            llks = np.empty(self.steps, dtype=np.float)
            llks[:] = np.nan
            return GenotypeTrace(genotypes, llks)

        # set the initial genotype
        if initial is None:
            # for each haplotype, take a random sample of mean of reads
            var_dist = _read_mean_dist(reads_het)
            genotype = np.empty((self.ploidy, n_het_base), dtype=np.int8)
            for i in range(self.ploidy):
                genotype[i] = probabilistic.sample_alleles(var_dist)
        else:
            # use the provided array
            assert initial.shape == (self.ploidy, n_het_base)
            genotype = initial.copy()

        # distribution to draw number of break points from for structural steps
        if self.n_intervals is None:
            # random number of intervals based on beta distribution
            break_dist = _point_beta_probabilities(n_het_base, self.alpha, self.beta)
        else:
            # this is a hack to fix number of intervals to a constant
            break_dist = np.zeros(self.n_intervals, dtype=np.float64)
            break_dist[-1] = 1  # 100% probability of n_intervals -1 break points

        # Automatically mask out alleles for which all reads have a probability of 0
        # A probability of 0 indicates that the allele is invalid (zero padding)
        # Masking alleles stops that allele being assessed in the
        # mcmc and hence reduces paramater space and compute time.
        mask = np.all(reads_het == 0, axis=-3)

        # run the sampler on the heterozygous positions
        genotypes, llks = _denovo_gibbs_sampler(
            genotype, 
            reads_het, 
            mask=mask,
            ratio=self.ratio, 
            steps=self.steps, 
            break_dist=break_dist,
            allow_recombinations=self.allow_recombinations,
            allow_dosage_swaps=self.allow_dosage_swaps,
            allow_deletions=self.allow_deletions
        )

        # return the genotype trace and llks
        if n_het_base == n_base:
            # no fixed homozygous alleles to add back in
            return GenotypeTrace(genotypes, llks)

        else:
            # add back in the fixed alleles that where removed
            # create a template of the fized alleles
            idx, vals = np.where(fixed)
            template = np.zeros(n_base, dtype=genotypes.dtype)
            template[idx] = vals
            # tile for each haplotype in each step
            template = np.tile(template, (self.steps, self.ploidy, 1))
            # add in the heterozygous alleles from the real trace
            template[:, :, heterozygous] = genotypes
            return GenotypeTrace(template, llks)
