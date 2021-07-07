import numpy as np
from numba import njit

from mchap.assemble.likelihood import log_genotype_prior
from mchap.assemble.util import genotype_alleles_as_index, index_as_genotype_alleles
from mchap.assemble.util import random_choice, normalise_log_probs

from .utils import allelic_dosage, count_allele
from .likelihood import log_likelihood_alleles_cached
from .prior import log_genotype_allele_prior


@njit(cache=True)
def compound_mh_step(
    genotype_alleles, haplotypes, reads, read_counts, inbreeding, llk_cache
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
def mh_mcmc(genotype_alleles, haplotypes, reads, read_counts, inbreeding, n_steps=1000):
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
    llk_cache = {}
    llk_cache[-1] = np.nan
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
    genotype_alleles, haplotypes, reads, read_counts, inbreeding, llk_cache
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
    genotype_alleles, haplotypes, reads, read_counts, inbreeding, n_steps=1000
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
    llk_cache = {}
    llk_cache[-1] = np.nan
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
def mcmc_posterior_phenotype_mode(genotype_alleles_trace):
    """Identify the mode genotype of the posterior mode allelic-phenotype.

    Parameters
    ----------
    genotype_alleles_trace : ndarray, int, shape (n_steps, ploidy)
        Genotype alleles trace.

    Returns
    -------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Alleles of the mode genotype of the posterior mode phenotype.
    genotype_probability
        Posterior probability of `genotype_alleles`.
    phenotype_probability
        Posterior probability of the mode phenotype.
    """
    n_steps, ploidy = genotype_alleles_trace.shape

    # mode phenotype
    phenotype_counts = {}
    for i in range(n_steps):
        gen = genotype_alleles_trace[i]
        phen = np.unique(gen)
        n = len(phen)
        idx = genotype_alleles_as_index(phen)
        key = (n, idx)
        if key not in phenotype_counts:
            phenotype_counts[key] = 1
        else:
            phenotype_counts[key] += 1
    mode_phen_key, mode_phen_count = (1, 0), 0
    for phen, count in phenotype_counts.items():
        if count > mode_phen_count:
            mode_phen_key, mode_phen_count = phen, count
    phenotype_prob = mode_phen_count / n_steps

    # mode genotype of phenotype
    genotype_counts = {}
    for i in range(n_steps):
        gen = genotype_alleles_trace[i]
        phen = np.unique(gen)
        n = len(phen)
        phen_idx = genotype_alleles_as_index(phen)
        if (n, phen_idx) == mode_phen_key:
            gen_idx = genotype_alleles_as_index(gen)
            if gen_idx in genotype_counts:
                genotype_counts[gen_idx] += 1
            else:
                genotype_counts[gen_idx] = 1
    mode_gen_idx, mode_gen_count = -1, 0
    for gen, count in genotype_counts.items():
        if count > mode_gen_count:
            mode_gen_idx, mode_gen_count = gen, count
    genotype = index_as_genotype_alleles(mode_gen_idx, ploidy)
    genotype_prob = mode_gen_count / n_steps

    return genotype, genotype_prob, phenotype_prob
