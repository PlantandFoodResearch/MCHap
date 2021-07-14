import numpy as np
from numba import njit

from mchap.assemble.likelihood import log_genotype_prior
from mchap.jitutils import random_choice, normalise_log_probs

from .utils import allelic_dosage, count_allele
from .likelihood import log_likelihood_alleles, log_likelihood_alleles_cached
from .prior import log_genotype_allele_prior


@njit(cache=True)
def mh_options(
    genotype_alleles,
    variable_allele,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    llks_array,
    lpriors_array,
    probabilities_array,
    llk_cache=None,
):
    """Calculate transition probabilities for a Metropolis-Hastings step.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    variable_allele : int
        Index of allele that is variable in this step.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.
    llks_array : ndarray, float , shape (n_haplotypes, )
        Array to be populated with log-likelihood of each step option.
    lpriors_array : ndarray, float , shape (n_haplotypes, )
        Array to be populated with log-prior of each step option.
    probabilities_array : ndarray, float , shape (n_haplotypes, )
        Array to be populated with transition probability of each step option.
    llk_cache : dict
        Cache of log-likelihoods mapping genotype index (int) to llk (float).

    Returns
    -------
    None

    Notes
    -----
    The `llks_array`, `lpriors_array`, and
    `probabilities_array` arrays are updated in place.

    """
    # save current allele
    current_allele = genotype_alleles[variable_allele]

    # stats of current genotype
    n_alleles = len(haplotypes)
    dosage = allelic_dosage(genotype_alleles)
    allele_copies = count_allele(genotype_alleles, genotype_alleles[variable_allele])
    lprior = log_genotype_prior(
        dosage=dosage,
        unique_haplotypes=n_alleles,
        inbreeding=inbreeding,
    )
    llk = log_likelihood_alleles_cached(
        reads=reads,
        read_counts=read_counts,
        haplotypes=haplotypes,
        genotype_alleles=genotype_alleles,
        cache=llk_cache,
    )

    # create array to hold proposal ratios for detailed balance
    lproposals_array = np.empty(n_alleles)

    # iterate over allele options
    for a in range(n_alleles):

        # handle case of current allele
        if genotype_alleles[variable_allele] == a:
            # proposed the same allele
            lproposals_array[a] = 0.0
            lpriors_array[a] = lprior
            llks_array[a] = llk

        # proposed new allele
        else:

            # set new allele
            genotype_alleles[variable_allele] = a

            # calculate prior
            dosage_i = allelic_dosage(genotype_alleles)
            lpriors_array[a] = log_genotype_prior(
                dosage=dosage_i,
                unique_haplotypes=n_alleles,
                inbreeding=inbreeding,
            )

            # calculate likelihood
            llks_array[a] = log_likelihood_alleles_cached(
                reads=reads,
                read_counts=read_counts,
                haplotypes=haplotypes,
                genotype_alleles=genotype_alleles,
                cache=llk_cache,
            )

            # calculate proposal ratio = g(G|G') / g(G'|G)
            allele_copies_i = count_allele(
                genotype_alleles, genotype_alleles[variable_allele]
            )
            lproposals_array[a] = np.log(allele_copies_i / allele_copies)

    # calculate transition ratios
    mh_ratio = (llks_array - llk) + (lpriors_array - lprior) + lproposals_array
    probabilities_array[:] = np.exp(np.minimum(0.0, mh_ratio))

    # calculate probability of no step
    probabilities_array[current_allele] = 0
    probabilities_array /= n_alleles - 1
    probabilities_array[current_allele] = 1 - probabilities_array.sum()

    # reset current allele
    genotype_alleles[variable_allele] = current_allele
    return None


@njit(cache=True)
def gibbs_options(
    genotype_alleles,
    variable_allele,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    llks_array,
    lpriors_array,
    probabilities_array,
    llk_cache=None,
):
    """Calculate transition probabilities for a Gibbs step.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Index of each haplotype in the genotype.
    variable_allele : int
        Index of allele that is variable in this step.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        Probabilistic reads.
    read_counts : ndarray, int, shape (n_reads, )
        Count of each read.
    inbreeding : float
        Expected inbreeding coefficient of sample.
    llks_array : ndarray, float , shape (n_haplotypes, )
        Array to be populated with log-likelihood of each step option.
    lpriors_array : ndarray, float , shape (n_haplotypes, )
        Array to be populated with log-prior of each step option.
    probabilities_array : ndarray, float , shape (n_haplotypes, )
        Array to be populated with transition probability of each step option.
    llk_cache : dict
        Cache of log-likelihoods mapping genotype index (int) to llk (float).

    Returns
    -------
    None

    Notes
    -----
    The `llks_array`, `lpriors_array`, and
    `probabilities_array` arrays are updated in place.

    """
    # save current allele
    current_allele = genotype_alleles[variable_allele]

    # iterate over allele options
    n_alleles = len(haplotypes)
    for a in range(n_alleles):

        # set genotype allele
        genotype_alleles[variable_allele] = a

        # genotype prior
        lpriors_array[a] = log_genotype_allele_prior(
            genotype=genotype_alleles,
            variable_allele=variable_allele,
            unique_haplotypes=len(haplotypes),
            inbreeding=inbreeding,
        )

        # genotype likelihood
        llks_array[a] = log_likelihood_alleles_cached(
            reads=reads,
            read_counts=read_counts,
            haplotypes=haplotypes,
            genotype_alleles=genotype_alleles,
            cache=llk_cache,
        )

    # gibbs step probabilities
    probabilities_array[:] = normalise_log_probs(llks_array + lpriors_array)

    # reset current allele
    genotype_alleles[variable_allele] = current_allele
    return None


@njit(cache=True)
def compound_step(
    genotype_alleles,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    llk_cache=None,
    step_type=0,
):
    """MCMC sampler compound step for calling sample alleles from a set of known genotypes.

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
    step_type : int
        Step type with 0 for a Gibbs-step and 1 for a
        Metropolis-Hastings step.

    Returns
    -------
    llk : float
        Log-likelihood of new genotype_alleles.

    Notes
    -----
    The `genotype_alleles` array is updated in place. This is a compound step
    in which samples each allele of the genotype in a random order.
    """
    ploidy = len(genotype_alleles)
    n_alleles = len(haplotypes)

    # arrays for proposed genotype stats
    lpriors = np.full(n_alleles, np.nan)
    llks = np.full(n_alleles, np.nan)
    probabilities = np.full(n_alleles, np.nan)

    # random random order of haplotypes of genotype
    order = np.arange(ploidy)
    np.random.shuffle(order)

    # iterate over every haplotype
    for j in range(ploidy):
        k = order[j]

        # calculate transition probabilities

        if step_type == 0:
            gibbs_options(
                genotype_alleles=genotype_alleles,
                variable_allele=k,
                haplotypes=haplotypes,
                reads=reads,
                read_counts=read_counts,
                inbreeding=inbreeding,
                llks_array=llks,
                lpriors_array=lpriors,
                probabilities_array=probabilities,
                llk_cache=llk_cache,
            )
        elif step_type == 1:
            mh_options(
                genotype_alleles=genotype_alleles,
                variable_allele=k,
                haplotypes=haplotypes,
                reads=reads,
                read_counts=read_counts,
                inbreeding=inbreeding,
                llks_array=llks,
                lpriors_array=lpriors,
                probabilities_array=probabilities,
                llk_cache=llk_cache,
            )
        else:
            raise ValueError("Unknown MCMC step type.")

        # make choice
        choice = random_choice(probabilities)

        # update genotype
        genotype_alleles[k] = choice

    # sort genotype and return llk of the final choice
    genotype_alleles.sort()
    return llks[choice]


@njit(cache=True)
def mcmc_sampler(
    genotype_alleles,
    haplotypes,
    reads,
    read_counts,
    inbreeding,
    n_steps=1000,
    cache=False,
    step_type=0,
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
    step_type : int
        Step type with 0 for a Gibbs-step and 1 for a
        Metropolis-Hastings step.

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
        llk = compound_step(
            genotype_alleles=genotype_alleles,
            haplotypes=haplotypes,
            reads=reads,
            read_counts=read_counts,
            inbreeding=inbreeding,
            llk_cache=llk_cache,
            step_type=step_type,
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
