#!/usr/bin/env python3

import numpy as np
import numba
from scipy import stats as _stats
from dataclasses import dataclass

from mchap.assemble import mutation, structural
from mchap.assemble.tempering import chain_swap_step
from mchap.assemble.likelihood import log_likelihood, new_log_likelihood_cache
from mchap.assemble import util
from mchap.assemble.classes import Assembler, GenotypeMultiTrace
from mchap.assemble.snpcalling import snp_posterior


__all__ = ["DenovoMCMC"]


@dataclass
class DenovoMCMC(Assembler):

    ploidy: int
    n_alleles: list
    inbreeding: float = 0
    steps: int = 1000
    chains: int = 2
    alpha: float = 1.0
    beta: float = 3.0
    n_intervals: int = None
    fix_homozygous: float = 0.999
    recombination_step_probability: float = 0.5
    partial_dosage_step_probability: float = 0.5
    dosage_step_probability: float = 1.0
    temperatures: tuple = (1.0,)
    random_seed: int = None
    llk_cache_threshold: int = 100
    """De novo haplotype assembly using Markov chain Monte Carlo
    for probabilistically encoded variable positions of NGS reads.

    Attributes
    ----------
    ploidy : int
        Ploidy of organism at the assembled locus.
    n_alleles : list, int
        Number of possible alleles at each position in the
        assembled locus.
    inbreeding : float
        Expected inbreeding coefficient of genotype.
    steps : int, optional
        Number of steps to run in each MCMC simulation
        (default = 1000).
    chains : int, optional
        Number of independent MCMC simulations to run
        (default = 2).
    alpha, beta : float, optional
        Parameters defining a Beta distribution to sample
        the number of random intervals to generate for each
        within-interval recombination and dosage step
        within the MCMC simulation.
        (defaults = 1.0, 3.0).
    n_intervals : int, optional
        If set structural steps in the MCMC will always use
        this number of intevals ignoring the alpha and beta
        parameters (default = None).
    fix_homozygous : float, optional
        Individual variant positions that have a posterior
        probability of being homozygous that is greater than
        this value will be fixed as non-variable positions
        during the MCMC simulation. This can greatly improve
        performance when there are multiple homozygous
        alleles (default = 0.999).
    recombination_step_probability : float, optional
        Probability of performing a recombination sub-step during
        each step of the MCMC.
        (default = 0.5).
    partial_dosage_step_probability : float, optional
        Probability of performing a within-interval dosage sub-step
        during each step of the MCMC.
        (default = 0.5).
    dosage_step_probability : float, optional
        Probability of performing a dosage sub-step during
        each step of the MCMC.
        (default = 1.0).
    temperatures : array_like float, optional
        Specify inverse temperatures for parallel tempering
        these should between 0 and 1 in ascending order with
        the final value being 1 (default = (1.0, )).
    random_seed: int, optional
        Seed the random seed for numpy and numba RNG
        (default = None).

    """

    def fit(self, reads, read_counts=None, initial=None):
        """Fit the parametized model to a set of probabilistically
        encoded variable positions of NGS reads.

        Parameters
        ----------
        reads : ndarray, float, shape (n_reads, n_positions, max_allele)
            Probabilistically encoded variable positions of NGS reads.
        read_counts : ndarray, int, shape (n_reads, )
            Optionally specify the number of observations of each read.
        initial : ndarray, int, shape (n_chains, ploidy, n_positions, max_allele), optional
            Set the initial genotype state of each MCMC simulation
            (default = None).

        Returns
        -------
        trace : GenotypeMultiTrace
            An instance of GenotypeMultiTrace containing the genotype state
            and log-likelihood at each step in each of the MCMC simulations.

        Notes
        -----
        If the initial genotype state is not set by the user
        then it is automatically set by sampling <ploidy> random
        haplotypes from the mean allele probabilities
        among all reads.

        """
        n_reads, n_pos, max_allele = reads.shape
        if n_reads == 0:
            # mock up a nan read
            assert len(self.n_alleles) == n_pos
            n_reads = 1
            reads = np.empty((n_reads, n_pos, max_allele), dtype=float)
            reads[:] = np.nan

        # set random seed once for all chains
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            util.seed_numba(self.random_seed)

        if initial is None:
            initial = [None for _ in range(self.chains)]

        # run simulations sequentially
        genotypes = []
        llks = []
        for chain in range(self.chains):
            gen_trace, llk_trace = self._mcmc(
                reads, read_counts=read_counts, initial=initial[chain]
            )
            genotypes.append(gen_trace)
            llks.append(llk_trace)

        # combine traces into multi-trace
        return GenotypeMultiTrace(
            np.array(genotypes),
            np.array(llks),
        )

    def _mcmc(self, reads, read_counts, initial=None):
        """Run a single MCMC simulation."""
        # identify base positions that are overwhelmingly likely
        # to be homozygous
        # these can be 'fixed' to reduce computational complexity
        hom_probs = _homozygosity_probabilities(
            reads,
            self.n_alleles,
            self.ploidy,
            inbreeding=self.inbreeding,
            read_counts=read_counts,
        )
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
            llks = np.empty(self.steps, dtype=float)
            llks[:] = np.nan
            return genotypes, llks

        # set the initial genotype
        if initial is None:
            dist = _read_mean_dist(reads_het)
            genotype = np.array([util.sample_alleles(dist) for _ in range(self.ploidy)])
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

        # only pass through allele numbers for heterozygous positions
        n_alleles = np.array(self.n_alleles, dtype=np.int8)
        assert len(n_alleles) == n_base
        n_alleles = n_alleles[heterozygous]

        # ensure temperatures is an ascending array ending in 1.0
        temperatures = np.sort(self.temperatures)
        assert temperatures[0] >= 0.0
        assert temperatures[-1] == 1.0

        # run the sampler on the heterozygous positions
        genotypes, llks = _denovo_gibbs_sampler(
            genotype=genotype,
            inbreeding=self.inbreeding,
            reads=reads_het,
            read_counts=read_counts,
            n_alleles=n_alleles,
            steps=self.steps,
            break_dist=break_dist,
            recombination_step_probability=self.recombination_step_probability,
            partial_dosage_step_probability=self.partial_dosage_step_probability,
            dosage_step_probability=self.dosage_step_probability,
            temperatures=temperatures,
            return_heated_trace=False,
            llk_cache_threshold=self.llk_cache_threshold,
        )

        # drop the first dimension of each trace component
        # this dimension is only used if heated chains are returned
        genotypes = genotypes[0]
        llks = llks[0]

        # return the genotype trace and llks
        if n_het_base == n_base:
            # no fixed homozygous alleles to add back in
            return genotypes, llks

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
            return template, llks


@numba.njit(cache=True)
def _denovo_gibbs_sampler(
    *,
    genotype,
    inbreeding,
    reads,
    read_counts,
    n_alleles,
    steps,
    break_dist,
    recombination_step_probability,
    partial_dosage_step_probability,
    dosage_step_probability,
    temperatures,
    return_heated_trace=False,
    llk_cache_threshold=100,
):
    """Gibbs sampler with parallele temporing"""
    # assert temperatures[-1] == 1.0
    ploidy, n_base = genotype.shape
    n_temps = len(temperatures)

    # number of possible unique haplotypes
    u_haps = np.prod(n_alleles)

    # one genotype state per temp
    genotypes = np.empty((n_temps, ploidy, n_base), dtype=genotype.dtype)
    for t in range(n_temps):
        genotypes[t] = genotype.copy()

    # likelihood for each genotype at each temp
    llks = np.empty(n_temps)
    llks[:] = log_likelihood(reads, genotype, read_counts=read_counts)

    # llk cache
    u_reads = len(reads)
    if ploidy * n_base * u_reads > llk_cache_threshold:
        cache = new_log_likelihood_cache(ploidy, n_base, max_alleles=np.max(n_alleles))
    else:
        cache = None

    if return_heated_trace:
        # trace for each chain
        genotype_trace = np.empty((n_temps, steps) + genotype.shape, np.int8)
        llk_trace = np.empty((n_temps, steps), np.float64)
    else:
        # trace for cold chain only
        genotype_trace = np.empty((1, steps) + genotype.shape, np.int8)
        llk_trace = np.empty((1, steps), np.float64)

    for i in range(steps):

        for t in range(n_temps):

            # get genotype and likelihood of state this temp
            llk = llks[t]
            genotype = genotypes[t]
            temp = temperatures[t]

            if np.isnan(llk):
                raise ValueError("Encountered log likelihood of nan")

            # mutation step
            llk, cache = mutation.compound_step(
                genotype=genotype,
                inbreeding=inbreeding,
                reads=reads,
                llk=llk,
                n_alleles=n_alleles,
                temp=temp,
                read_counts=read_counts,
                cache=cache,
            )

            # recombinations step
            if np.random.rand() <= recombination_step_probability:
                n_breaks = util.random_choice(break_dist)
                intervals = structural.random_breaks(n_breaks, n_base)
                llk, cache = structural.compound_step(
                    genotype=genotype,
                    inbreeding=inbreeding,
                    reads=reads,
                    llk=llk,
                    intervals=intervals,
                    n_alleles=n_alleles,
                    step_type=0,
                    temp=temp,
                    read_counts=read_counts,
                    cache=cache,
                )

            # interval dosage step
            if np.random.rand() <= partial_dosage_step_probability:
                n_breaks = util.random_choice(break_dist)
                intervals = structural.random_breaks(n_breaks, n_base)
                llk, cache = structural.compound_step(
                    genotype=genotype,
                    inbreeding=inbreeding,
                    reads=reads,
                    llk=llk,
                    intervals=intervals,
                    n_alleles=n_alleles,
                    step_type=1,
                    temp=temp,
                    read_counts=read_counts,
                    cache=cache,
                )

            # final full length dosage swap
            if np.random.rand() <= dosage_step_probability:
                llk, cache = structural.compound_step(
                    genotype=genotype,
                    inbreeding=inbreeding,
                    reads=reads,
                    llk=llk,
                    intervals=np.array([[0, n_base]]),
                    n_alleles=n_alleles,
                    step_type=1,
                    temp=temp,
                    read_counts=read_counts,
                    cache=cache,
                )

            # chain swap step if not the highest temp
            if t > 0:
                llk_prev = llks[t - 1]
                genotype_prev = genotypes[t - 1]
                temp_prev = temperatures[t - 1]
                llk, llk_prev = chain_swap_step(
                    genotype_i=genotype,
                    llk_i=llk,
                    temp_i=temp,
                    genotype_j=genotype_prev,
                    llk_j=llk_prev,
                    temp_j=temp_prev,
                    inbreeding=inbreeding,
                    unique_haplotypes=u_haps,
                )
                llks[t - 1] = llk_prev

            # save llk of current temp
            llks[t] = llk

            # save llk of current temp
            llks[t] = llk

        if return_heated_trace:
            # save state and likelihood of each chain
            genotype_trace[:, i] = genotypes.copy()
            llk_trace[:, i] = llks.copy()
        else:
            # save state and likelihood of cold chain only
            genotype_trace[0, i] = genotype.copy()
            llk_trace[0, i] = llk
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
    """Calculate the element-wise means of a collection of
    probabilistically encoded reads.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_positions, max_allele)
        Probabilistically encoded variable positions of NGS reads.

    Returns
    -------
    mean_read : ndarray, float, shape (n_positions, max_allele)
        The mean probabilities of input reads.

    Notes
    -----
    If read distributions are normalized if they do not sum to 1.

    """
    # work around to avoid nan values caused by gaps
    reads = reads.copy()
    n_reads = len(reads)
    gaps = np.isnan(reads).all(axis=0)

    # replace gaps with 1
    reads[np.tile(gaps, (n_reads, 1, 1))] = 1
    dist = np.nanmean(reads, axis=0)

    # fill gaps
    n_alleles = np.sum(~np.all(reads == 0, axis=0), axis=1, keepdims=True)
    fill = 1 / np.tile(n_alleles, (1, reads.shape[-1]))
    dist[gaps] = fill[gaps]

    # normalize
    dist /= dist.sum(axis=-1, keepdims=True)

    return dist


def _homozygosity_probabilities(
    reads, n_alleles, ploidy, inbreeding=0, read_counts=None
):
    """Calculate posterior probabilities at each single SNP position to determine
    if an individual is homozygous for a single allele.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_positions, max_allele)
        Reads encoded as probability distributions.
    n_alleles : array_like, int, shape(n_positions, )
        Number of possible alleles for each SNP.
    ploidy : int
        Ploidy of organism.
    inbreeding : float
        Expected inbreeding coefficient of organism.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.

    Returns
    -------
    homozygosity_probs : ndarray, float, shape (n_positions, max_allele)
        Probability of each homozygous genotype for each SNP

    Notes
    -----
    The probabilities calculated this way are independent
    of alleles at other positions.
    """
    _, n_pos, max_allele = reads.shape
    probabilites = np.zeros((n_pos, max_allele), dtype=float)

    for i in range(n_pos):
        n = n_alleles[i]

        # calculate posterior distribution
        genotypes, probs = snp_posterior(
            reads, i, n, ploidy, inbreeding, read_counts=read_counts
        )

        # look at homozygous genotypes
        homozygous = np.all(genotypes[:, 0:1] == genotypes, axis=-1)

        # these are in same order as the hom allele
        hom_probs = probs[homozygous]
        probabilites[i, 0:n] = hom_probs

    return probabilites
