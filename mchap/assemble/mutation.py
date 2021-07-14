#!/usr/bin/env python3

import numpy as np
import numba

from mchap.jitutils import random_choice, count_haplotype_copies, get_haplotype_dosage
from mchap.assemble.likelihood import log_likelihood_cached, log_genotype_prior


__all__ = ["base_step", "compound_step"]


@numba.njit(cache=True)
def base_step(
    genotype,
    reads,
    llk,
    h,
    j,
    unique_haplotypes,
    inbreeding=0,
    n_alleles=None,
    temp=1,
    read_counts=None,
    cache=None,
):
    """Mutation Gibbs sampler step for the jth base position
    of the hth haplotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded
        as simple integers from 0 to n_nucl.
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic
        matrices.
    llk : float
        Log-likelihood of the initial haplotype state given
        the observed reads.
    h : int
        Index of the haplotype to be mutated.
    j : int
        Index of the base position to be mutated
    unique_haplotypes : int
        Total number of unique haplotypes possible at this locus.
    inbreeding : float
        Expected inbreeding coefficient of the genotype.
    n_alleles : int
        Number of possible base alleles at this positions.
    temp : float
        An inverse temperature in the interval 0, 1 to adjust
        the sampled distribution by.
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.
    cache : tuple
        An array_map tuple created with `new_log_likelihood_cache` or None.

    Returns
    -------
    llk : float
        New log-likelihood of observed reads given
        the updated genotype.
    cache_updated : tuple
        Updated cache or None.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    assert 0 <= temp <= 1
    ploidy = len(genotype)
    # number of possible alleles given given array size
    if n_alleles is None:
        n_alleles = reads.shape[-1]

    # store log likelihoods calculated with each allele
    llks = np.empty(n_alleles)

    # store MH acceptance probability for each allele
    log_accept = np.empty(n_alleles)

    # use differences in count of haplotype h to calculate
    # ratio of proposal probabilities
    lhapcount = np.log(count_haplotype_copies(genotype, h))

    # ratio of prior probabilities
    dosage = np.empty(ploidy, dtype=np.int8)
    get_haplotype_dosage(dosage, genotype)
    lprior = log_genotype_prior(dosage, unique_haplotypes, inbreeding)

    current_nucleotide = genotype[h, j]
    n_options = 0
    for i in range(n_alleles):
        if i == current_nucleotide:
            # store current likelihood
            llks[i] = llk
            log_accept[i] = -np.inf  # log(0)
        else:
            # count number of possible new genotypes
            n_options += 1

            # set the current genotype to new genotype
            genotype[h, j] = i

            # calculate and store log-likelihood: P(G'|R)
            llk_i, cache = log_likelihood_cached(
                reads, genotype, cache=cache, read_counts=read_counts
            )
            llks[i] = llk_i

            # calculate log likelihood ratio: ln(P(G'|R)/P(G|R))
            llk_ratio = llk_i - llk

            # calculate ratio of priors: ln(P(G')/P(G))
            get_haplotype_dosage(dosage, genotype)
            lprior_i = log_genotype_prior(dosage, unique_haplotypes, inbreeding)
            lprior_ratio = lprior_i - lprior

            # calculate proposal ratio for detailed balance: ln(g(G|G')/g(G'|G))
            lhapcount_i = np.log(count_haplotype_copies(genotype, h))
            lproposal_ratio = lhapcount_i - lhapcount

            # calculate Metropolis-Hastings acceptance probability
            # ln(min(1, (P(G'|R)P(G')g(G|G')) / (P(G|R)P(G)g(G'|G)))
            mh_ratio = (llk_ratio + lprior_ratio) * temp + lproposal_ratio
            log_accept[i] = np.minimum(0.0, mh_ratio)  # max prob of log(1)

    # divide acceptance probability by number of steps to choose from
    log_accept -= np.log(n_options)

    # convert to probability of proposal * probability of acceptance
    # then fill in probability that no step is made (i.e. choose the initial state)
    probabilities = np.exp(log_accept)
    probabilities[current_nucleotide] = 1 - probabilities.sum()

    # random choice of new state using probabilities
    choice = random_choice(probabilities)

    # update state
    genotype[h, j] = choice

    # return final log liklihood
    return llks[choice], cache


@numba.njit(cache=True)
def compound_step(
    genotype,
    reads,
    llk,
    inbreeding=0,
    n_alleles=None,
    temp=1,
    read_counts=None,
    cache=None,
):
    """Mutation compound Gibbs sampler step for all base positions
    of all haplotypes in a genotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as
        simple integers from 0 to n_nucl.
    reads : ndarray, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    llk : float
        Log-likelihood of the initial haplotype state given the
        observed reads.
    inbreeding : float
        Expected inbreeding coefficient of the genotype.
    n_alleles : ndarray, int, shape (n_base, )
        The number of possible alleles at each base position.
    temp : float
        An inverse temperature in the interval 0, 1 to adjust
        the sampled distribution by.
    read_counts : ndarray, int, shape (n_reads, )
        Optionally specify the number of observations of
        each read.
    cache : tuple
        An array_map tuple created with `new_log_likelihood_cache` or None.

    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype.
    cache_updated : tuple
        Updated cache or None.

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    ploidy, n_base = genotype.shape

    if n_alleles is None:
        max_allele = reads.shape[-1]
        n_alleles = np.empty(n_base, dtype=np.int8)
        n_alleles[:] = max_allele
        unique_haplotypes = n_base ** max_allele
    else:
        unique_haplotypes = np.prod(n_alleles)

    # matrix of haplotype-base combinations
    substeps = np.empty((ploidy * n_base, 2), dtype=np.int8)

    for h in range(ploidy):
        for j in range(n_base):
            substep = (h * n_base) + j
            substeps[substep, 0] = h
            substeps[substep, 1] = j

    # random order
    np.random.shuffle(substeps)

    for i in range(ploidy * n_base):
        h, j = substeps[i]
        llk, cache = base_step(
            genotype=genotype,
            reads=reads,
            llk=llk,
            h=h,
            j=j,
            cache=cache,
            unique_haplotypes=unique_haplotypes,
            inbreeding=inbreeding,
            n_alleles=n_alleles[j],
            temp=temp,
            read_counts=read_counts,
        )
    return llk, cache
