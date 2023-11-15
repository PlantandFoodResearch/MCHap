import numpy as np
from numba import njit

from mchap.jitutils import random_choice, normalise_log_probs
from mchap.calling.utils import count_allele
from .likelihood import log_likelihood_alleles_cached
from .prior import markov_blanket_log_probability, markov_blanket_log_allele_probability


@njit(cache=True)
def metropolis_hastings_probabilities(
    target_index,
    allele_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
    sample_read_counts,  # array (n_samples, n_reads)
    haplotypes,  # (n_haplotypes, n_pos)
    log_frequencies,  # (n_haplotypes,)
    llk_cache,
):
    n_alleles = len(haplotypes)
    ploidy = sample_ploidy[target_index]
    current_allele = sample_genotypes[target_index, allele_index]
    allele_copies = count_allele(sample_genotypes[target_index], current_allele)
    reads = sample_read_dists[target_index]
    read_counts = sample_read_counts[target_index]
    idx = read_counts > 0
    reads = reads[idx]
    read_counts = read_counts[idx]

    # current likelihood and prior
    llk = log_likelihood_alleles_cached(
        reads=reads,
        read_counts=read_counts,
        haplotypes=haplotypes,
        sample=target_index,
        genotype_alleles=np.sort(sample_genotypes[target_index, 0:ploidy]),
        cache=llk_cache,
    )
    lprior = markov_blanket_log_probability(
        target_index=target_index,
        sample_genotypes=sample_genotypes,
        sample_ploidy=sample_ploidy,
        sample_parents=sample_parents,
        gamete_tau=gamete_tau,
        gamete_lambda=gamete_lambda,
        gamete_error=gamete_error,
        log_frequencies=log_frequencies,
    )

    # store MH acceptance probability for each allele
    log_accept = np.empty(n_alleles)

    for i in range(n_alleles):
        if i == current_allele:

            # store current likelihood
            log_accept[i] = -np.inf  # log(0)
        else:
            sample_genotypes[target_index, allele_index] = i

            # calculate log likelihood ratio: ln(P(G'|R)/P(G|R))
            llk_i = log_likelihood_alleles_cached(
                reads=reads,
                read_counts=read_counts,
                haplotypes=haplotypes,
                sample=target_index,
                genotype_alleles=np.sort(sample_genotypes[target_index, 0:ploidy]),
                cache=llk_cache,
            )
            llk_ratio = llk_i - llk

            # calculate ratio of priors: ln(P(G')/P(G))
            lprior_i = markov_blanket_log_probability(
                target_index=target_index,
                sample_genotypes=sample_genotypes,
                sample_ploidy=sample_ploidy,
                sample_parents=sample_parents,
                gamete_tau=gamete_tau,
                gamete_lambda=gamete_lambda,
                gamete_error=gamete_error,
                log_frequencies=log_frequencies,
            )
            lprior_ratio = lprior_i - lprior

            # calculate proposal ratio = g(G|G') / g(G'|G)
            allele_copies_i = count_allele(sample_genotypes[target_index], i)
            lproposal_ratio = np.log(allele_copies_i / allele_copies)

            # calculate Metropolis-Hastings acceptance probability
            # ln(min(1, (P(G'|R)P(G')g(G|G')) / (P(G|R)P(G)g(G'|G)))
            log_accept[i] = np.minimum(
                0.0, llk_ratio + lprior_ratio + lproposal_ratio
            )  # max prob of log(1)

    # adjust acceptance probabilities by proposal probabilities
    log_accept -= np.log(n_alleles - 1)

    # convert to probability of proposal * probability of acceptance
    # then fill in probability that no step is made (i.e. choose the initial state)
    probabilities = np.exp(log_accept)
    probabilities[current_allele] = 1 - probabilities.sum()

    # reset current allele
    sample_genotypes[target_index, allele_index] = current_allele

    return probabilities


@njit(cache=True)
def gibbs_probabilities(
    target_index,
    allele_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
    sample_read_counts,  # array (n_samples, n_reads)
    haplotypes,  # (n_haplotypes, n_pos)
    log_frequencies,  # (n_haplotypes,)
    llk_cache,
):
    n_alleles = len(haplotypes)
    ploidy = sample_ploidy[target_index]
    current_allele = sample_genotypes[target_index, allele_index]
    reads = sample_read_dists[target_index]
    read_counts = sample_read_counts[target_index]
    idx = read_counts > 0
    reads = reads[idx]
    read_counts = read_counts[idx]

    # store gibbs probability for each allele
    log_probabilities = np.empty(n_alleles)

    for i in range(n_alleles):
        sample_genotypes[target_index, allele_index] = i

        # calculate log likelihood ratio: ln(P(G'|R)/P(G|R))
        llk_i = log_likelihood_alleles_cached(
            reads=reads,
            read_counts=read_counts,
            haplotypes=haplotypes,
            sample=target_index,
            genotype_alleles=np.sort(sample_genotypes[target_index, 0:ploidy]),
            cache=llk_cache,
        )

        # calculate ratio of priors: ln(P(G')/P(G))
        lprior_i = markov_blanket_log_allele_probability(
            target_index=target_index,
            allele_index=allele_index,
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            log_frequencies=log_frequencies,
        )
        log_probabilities[i] = llk_i + lprior_i

    probabilities = normalise_log_probs(log_probabilities)
    # reset current allele
    sample_genotypes[target_index, allele_index] = current_allele
    return probabilities


@njit(cache=True)
def allele_step(
    target_index,
    allele_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
    sample_read_counts,  # array (n_samples, n_reads)
    haplotypes,  # (n_haplotypes, n_pos)
    log_frequencies,
    llk_cache,
    step_type=0,  # 0=Gibbs, 1=MH
):
    if step_type == 0:
        probabilities = gibbs_probabilities(
            target_index=target_index,
            allele_index=allele_index,
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
        )
    elif step_type == 1:
        probabilities = metropolis_hastings_probabilities(
            target_index=target_index,
            allele_index=allele_index,
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
        )
    else:
        raise ValueError
    # random choice of new state using probabilities
    choice = random_choice(probabilities)
    sample_genotypes[target_index, allele_index] = choice


@njit(cache=True)
def sample_step(
    target_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,
    sample_read_counts,
    haplotypes,
    log_frequencies,
    llk_cache,
    step_type=0,
):
    allele_indices = np.arange(sample_ploidy[target_index])
    np.random.shuffle(allele_indices)
    for i in range(len(allele_indices)):
        allele_step(
            target_index=target_index,
            allele_index=allele_indices[i],
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            step_type=step_type,
        )


@njit(cache=True)
def compound_step(
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,
    sample_read_counts,
    haplotypes,
    log_frequencies,
    llk_cache,
    step_type=0,
):
    target_indices = np.arange(len(sample_genotypes))
    np.random.shuffle(target_indices)
    for i in range(len(target_indices)):
        sample_step(
            target_index=target_indices[i],
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            step_type=step_type,
        )


@njit(cache=True)
def mcmc_sampler(
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,
    sample_read_counts,
    haplotypes,
    log_frequencies,
    n_steps=2000,
    annealing=1000,
    step_type=0,
):
    """MCMC simulation for calling alleles in pedigreed genotypes from a set of known haplotypes.

    Parameters
    ----------
    sample_genotypes : ndarray, int, shape (n_samples, ploidy)
        Index of each haplotype in each genotype for each sample.
    sample_ploidy : ndarray, int, shape (n_samples,)
        Ploidy of each samples
    sample_parents : ndarray, int, shape (n_samples, 2)
        Indices of the parents of each sample with negative values
        indicating unknown parents.
    gamete_tau : ndarray, int, shape (n_samples, 2)
        Number of chromosomal copies contributed by each parent.
    gamete_lambda : ndarray, float, shape (n_samples, 2)
        Excess IBD caused by miotic processes.
    gamete_error : ndarray, float, shape (n_samples, 2)
        Error term associated with each gamete.
    sample_read_dists : ndarray, float, shape (n_samples, n_reads, n_pos, n_nucl)
        Probabilistically encoded reads for each samples.
    sample_read_counts : ndarray, int, shape (n_samples, n_reads)
        Number of observations of each read for each samples.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes.
    log_frequencies : ndarray, float, shape (n_haplotypes,)
        Log of prior frequencies for haplotypes.
    n_steps : int
        Number of (compound) steps to simulate.
    annealing : int
        Number of initial steps in which to perform simulated annealing.
    step_type : int
        Step type with 0 for a Gibbs-step and 1 for a
        Metropolis-Hastings step.

    Notes
    -----
    The gamete_lambda variable only supports non-zero values when
    gametic ploidy (tau) is 2.

    Returns
    -------
    genotype_alleles_trace : ndarray, int, shape (n_samples, n_steps, ploidy)
        Genotype alleles trace for each sample.
    """
    # set up caches
    llk_cache = {}
    llk_cache[(-1, -1)] = np.nan

    # sample error weighting for annealing burin in
    error_weight = np.ones(n_steps, np.float64)
    if annealing:
        error_weight[0:annealing] = np.linspace(0.0, 1.0, annealing)

    sample_genotypes = sample_genotypes.copy()
    n_samples, max_ploidy = sample_genotypes.shape
    trace = np.empty((n_steps, n_samples, max_ploidy), dtype=sample_genotypes.dtype)
    for i in range(n_steps):
        compound_step(
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            step_type=step_type,
        )
        trace[i] = sample_genotypes.copy()

    # sort trace allowing for mixed ploidy
    for j in range(n_samples):
        ploidy = sample_ploidy[j]
        for i in range(n_steps):
            trace[i, j] = np.sort(trace[i, j])
            if ploidy < max_ploidy:
                trace[i, j] = np.roll(trace[i, j], ploidy - max_ploidy)
    return trace
