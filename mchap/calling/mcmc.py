import numpy as np
from numba import njit

from mchap.assemble.likelihood import log_genotype_prior
from mchap.assemble.util import random_choice, normalise_log_probs

from .utils import allelic_dosage, count_allele
from .likelihood import log_likelihood_alleles, log_likelihood_alleles_cached
from .prior import log_genotype_allele_prior


@njit(cache=True)
def compound_mh_step(
    genotype_alleles, haplotypes, reads, read_counts, inbreeding, llk_cache=None
):
    """Metropolis-Hastings MCMC compound step for calling sample alleles from a set of known genotypes.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.
    llk_cache : dict
        Cache of log-likelihoods mapping genotype index (int) to llk (float).

    Returns
    -------
    llk : float
        Log-likelihood of new genotype_alleles.

    Notes
    -----
    The `genotype_alleles` array is updated in place.
    """
    # alias genotype_alleles
    genotype = genotype_alleles
    ploidy = len(genotype)
    n_alleles = len(haplotypes)

    # stats of current genotype
    dosage = allelic_dosage(genotype)
    lprior = log_genotype_prior(
        dosage=dosage,
        unique_haplotypes=n_alleles,
        inbreeding=inbreeding,
    )
    llk = log_likelihood_alleles_cached(
        reads=reads,
        read_counts=read_counts,
        haplotypes=haplotypes,
        genotype_alleles=genotype,
        cache=llk_cache,
    )

    # array for proposed genotype
    genotype_i = genotype.copy()

    # arrays for proposed genotype stats
    llproposals = np.full(n_alleles, np.nan)
    lpriors = np.full(n_alleles, np.nan)
    llks = np.full(n_alleles, np.nan)

    # iterate over every haplotype
    for k in range(ploidy):
        allele_copies = count_allele(genotype, genotype[k])
        # iterate of ever possible allele
        for a in range(n_alleles):

            if genotype[k] == a:
                # proposed the same allele
                llproposals[a] = 0.0
                lpriors[a] = lprior
                llks[a] = llk
            else:
                # stats of proposed genotype
                genotype_i[k] = a
                dosage_i = allelic_dosage(genotype_i)
                lpriors[a] = log_genotype_prior(
                    dosage=dosage_i,
                    unique_haplotypes=n_alleles,
                    inbreeding=inbreeding,
                )
                llks[a] = log_likelihood_alleles_cached(
                    reads=reads,
                    read_counts=read_counts,
                    haplotypes=haplotypes,
                    genotype_alleles=genotype_i,
                    cache=llk_cache,
                )
                # proposal ratio = g(G|G') / g(G'|G)
                allele_copies_i = count_allele(genotype_i, genotype_i[k])
                llproposals[a] = np.log(allele_copies_i / allele_copies)

        # calculate transition ratios
        mh_ratio = (llks - llk) + (lpriors - lprior) + llproposals
        acceptence_probs = np.exp(np.minimum(0.0, mh_ratio))
        # zero current genotype probability
        acceptence_probs[genotype[k]] = 0
        # divide by number of options
        acceptence_probs /= n_alleles - 1
        # chance of no step
        acceptence_probs[genotype[k]] = 1 - acceptence_probs.sum()

        # make choice
        choice = random_choice(acceptence_probs)

        # print(genotype, genotype[k], choice)
        # print(acceptence_probs, 5)
        # print("\n")

        if choice == genotype[k]:
            # no step
            genotype_i[k] = choice
        else:
            genotype[k] = choice
            genotype_i[k] = choice
            llk = llks[choice]
            lprior = lpriors[choice]
            dosage = allelic_dosage(genotype)

    genotype.sort()
    return llk


@njit(cache=True)
def mh_mcmc(
    genotype_alleles,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    n_steps=1000,
    cache=False,
):
    """MCMC simulation for calling sample alleles from a set of known genotypes.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.
    n_steps : int
        Number of (compound) steps to simulate.

    Returns
    -------
    genotype_alleles_trace : ndarray, int, shape (n_steps, ploidy)
        Genotype alleles trace.
    llk_trace : ndarray, float, shape (n_steps, )
        Log-likelihood of new genotype_alleles.

    """
    genotype_alleles = genotype_alleles.copy()
    ploidy = len(genotype_alleles)
    genotype_trace = np.empty((n_steps, ploidy), genotype_alleles.dtype)
    llk_trace = np.empty(n_steps, np.float64)
    if cache:
        llk_cache = {}
        llk_cache[-1] = np.nan
    else:
        llk_cache = None
    for i in range(n_steps):
        llk = compound_mh_step(
            genotype_alleles=genotype_alleles,
            haplotypes=haplotypes,
            reads=reads,
            read_counts=read_counts,
            inbreeding=inbreeding,
            llk_cache=llk_cache,
        )
        llk_trace[i] = llk
        genotype_trace[i] = genotype_alleles.copy()
    return genotype_trace, llk_trace


@njit(cache=True)
def compound_gibbs_step(
    genotype_alleles,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    llk_cache=None,
):
    """Gibbs sampler step for calling sample alleles from a set of known genotypes.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.
    llk_cache : dict
        Cache of log-likelihoods mapping genotype index (int) to llk (float).

    Returns
    -------
    llk : float
        Log-likelihood of new genotype_alleles.

    Notes
    -----
    The `genotype_alleles` array is updated in place. This is a compound step
    in which samples each allele of the genotype in a random order.
    """
    # alias genotype_alleles
    genotype = genotype_alleles
    ploidy = len(genotype)
    n_alleles = len(haplotypes)

    # array for proposed genotype
    genotype_i = genotype.copy()

    # arrays for proposed genotype stats
    lpriors = np.full(n_alleles, np.nan)
    llks = np.full(n_alleles, np.nan)

    # random random order of haplotypes of genotype
    order = np.arange(ploidy)
    np.random.shuffle(order)

    # iterate over every haplotype
    for j in range(ploidy):
        k = order[j]

        # iterate of ever possible allele
        for a in range(n_alleles):

            # stats of proposed genotype
            genotype_i[k] = a

            lpriors[a] = log_genotype_allele_prior(
                genotype=genotype_i,
                variable_allele=k,
                unique_haplotypes=len(haplotypes),
                inbreeding=inbreeding,
            )
            llks[a] = log_likelihood_alleles_cached(
                reads=reads,
                read_counts=read_counts,
                haplotypes=haplotypes,
                genotype_alleles=genotype_i,
                cache=llk_cache,
            )

        # calculate gibbs transition probabilities
        gibbs_probabilities = normalise_log_probs(llks + lpriors)

        # make choice
        choice = random_choice(gibbs_probabilities)

        # update genotype and reset proposed genotype
        genotype[k] = choice
        genotype_i[k] = choice

    # sort genotype and return llk of the final choice
    genotype.sort()
    return llks[choice]


@njit(cache=True)
def gibbs_mcmc(
    genotype_alleles,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    n_steps=1000,
    cache=False,
):
    """Gibbs sampler MCMC simulation for calling sample alleles from a set of known genotypes.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.
    n_steps : int
        Number of (compound) steps to simulate.

    Returns
    -------
    genotype_alleles_trace : ndarray, int, shape (n_steps, ploidy)
        Genotype alleles trace.
    llk_trace : ndarray, float, shape (n_steps, )
        Log-likelihood of new genotype_alleles.

    """
    genotype_alleles = genotype_alleles.copy()
    ploidy = len(genotype_alleles)
    genotype_trace = np.empty((n_steps, ploidy), genotype_alleles.dtype)
    llk_trace = np.empty(n_steps, np.float64)
    if cache:
        llk_cache = {}
        llk_cache[-1] = np.nan
    else:
        llk_cache = None
    for i in range(n_steps):
        llk = compound_gibbs_step(
            genotype_alleles=genotype_alleles,
            haplotypes=haplotypes,
            reads=reads,
            read_counts=read_counts,
            inbreeding=inbreeding,
            llk_cache=llk_cache,
        )
        llk_trace[i] = llk
        genotype_trace[i] = genotype_alleles.copy()
    return genotype_trace, llk_trace


@njit(cache=True)
def greedy_caller(haplotypes, ploidy, reads, read_counts, inbreeding=0.0):
    """Greedy method for calling genotype from known haplotypes.

    Parameters
    ----------
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    ploidy : int
        Ploidy of organism locus.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.

    Returns
    -------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.

    """
    n_alleles = len(haplotypes)
    previous_genotype = np.zeros(0, np.int8)
    for i in range(ploidy):
        # add new allele slot to genotype
        k = i + 1
        genotype = np.zeros(k, np.int8)
        # copy alleles from previous loop
        genotype[0:i] = previous_genotype[0:i]

        best_lprob = -np.inf
        best_allele = -1
        for a in range(n_alleles):
            genotype[i] = a
            llk = log_likelihood_alleles(
                reads=reads,
                read_counts=read_counts,
                haplotypes=haplotypes,
                genotype_alleles=genotype,
            )
            lprior = log_genotype_prior(
                dosage=allelic_dosage(genotype),
                unique_haplotypes=len(haplotypes),
                inbreeding=inbreeding,
            )
            lprob = llk + lprior
            if lprob > best_lprob:
                # update best
                best_lprob = lprob
                best_allele = a
        # greedy choice
        genotype[i] = best_allele
        previous_genotype = genotype
    genotype.sort()
    return genotype
