import numpy as np
from numba import njit

from mchap.jitutils import random_choice, normalise_log_probs
from mchap.calling.utils import count_allele
from .likelihood import log_likelihood_alleles_cached
from .prior import (
    markov_blanket_log_probability,
    markov_blanket_log_allele_probability,
    generic_markov_blanket_log_probability,
)


@njit(cache=True)
def metropolis_hastings_probabilities(
    target_index,
    allele_index,
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
    sample_read_counts,  # array (n_samples, n_reads)
    haplotypes,  # (n_haplotypes, n_pos)
    log_frequencies,  # (n_haplotypes,)
    llk_cache,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
    dosage_log_frequencies,
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
        sample_children=sample_children,
        gamete_tau=gamete_tau,
        gamete_lambda=gamete_lambda,
        gamete_error=gamete_error,
        log_frequencies=log_frequencies,
        dosage=dosage,
        dosage_p=dosage_p,
        dosage_q=dosage_q,
        gamete_p=gamete_p,
        gamete_q=gamete_q,
        constraint_p=constraint_p,
        constraint_q=constraint_q,
        dosage_log_frequencies=dosage_log_frequencies,
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
                sample_children=sample_children,
                gamete_tau=gamete_tau,
                gamete_lambda=gamete_lambda,
                gamete_error=gamete_error,
                log_frequencies=log_frequencies,
                dosage=dosage,
                dosage_p=dosage_p,
                dosage_q=dosage_q,
                gamete_p=gamete_p,
                gamete_q=gamete_q,
                constraint_p=constraint_p,
                constraint_q=constraint_q,
                dosage_log_frequencies=dosage_log_frequencies,
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
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
    sample_read_counts,  # array (n_samples, n_reads)
    haplotypes,  # (n_haplotypes, n_pos)
    log_frequencies,  # (n_haplotypes,)
    llk_cache,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
    dosage_log_frequencies,
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
            sample_children=sample_children,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            log_frequencies=log_frequencies,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
            dosage_log_frequencies=dosage_log_frequencies,
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
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
    sample_read_counts,  # array (n_samples, n_reads)
    haplotypes,  # (n_haplotypes, n_pos)
    log_frequencies,
    llk_cache,
    step_type,  # 0=Gibbs, 1=MH
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
    dosage_log_frequencies,
):
    if step_type == 0:
        probabilities = gibbs_probabilities(
            target_index=target_index,
            allele_index=allele_index,
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            sample_children=sample_children,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
            dosage_log_frequencies=dosage_log_frequencies,
        )
    elif step_type == 1:
        probabilities = metropolis_hastings_probabilities(
            target_index=target_index,
            allele_index=allele_index,
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            sample_children=sample_children,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
            dosage_log_frequencies=dosage_log_frequencies,
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
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,
    sample_read_counts,
    haplotypes,
    log_frequencies,
    llk_cache,
    step_type,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
    dosage_log_frequencies,
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
            sample_children=sample_children,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            step_type=step_type,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
            dosage_log_frequencies=dosage_log_frequencies,
        )


@njit(cache=True)
def compound_step(
    sample_genotypes,
    sample_ploidy,
    sample_parents,
    sample_children,
    gamete_tau,
    gamete_lambda,
    gamete_error,
    sample_read_dists,
    sample_read_counts,
    haplotypes,
    log_frequencies,
    llk_cache,
    step_type,
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
    dosage_log_frequencies,
):
    target_indices = np.arange(len(sample_genotypes))
    np.random.shuffle(target_indices)
    for i in range(len(target_indices)):
        sample_step(
            target_index=target_indices[i],
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            sample_children=sample_children,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            step_type=step_type,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
            dosage_log_frequencies=dosage_log_frequencies,
        )


@njit(cache=True)
def sample_children_matrix(sample_parents):
    """Identify the children of each sample

    Parameters
    ----------
    sample_parents : ndarray, int, shape (n_samples, 2)
        Integer indices of parents with -1 indicating unknown.

    Returns
    -------
    sample_children : ndarray, int, shape (n_samples, max_children)
        Integer indices of children padded by -1.
    """
    n_samples, n_parents = sample_parents.shape
    assert n_parents == 2
    next_child_index = np.zeros(n_samples, dtype=np.int64)
    # first loop through is to count the number of children
    # this is 2x iteration but avoids the creation of an n*n matrix
    for i in range(n_samples):
        for j in range(n_parents):
            p = sample_parents[i, j]
            assert p != i  # can't be your own parent!
            if p >= 0:
                if j == 1:
                    # check for selfing
                    if p == sample_parents[i, 0]:
                        break
                next_child_index[p] += 1
    max_children = next_child_index.max()
    sample_children = np.full((n_samples, max_children), -1, dtype=np.int64)
    next_child_index[:] = 0
    for i in range(n_samples):
        for j in range(n_parents):
            p = sample_parents[i, j]
            if p >= 0:
                if j == 1:
                    # check for selfing
                    if p == sample_parents[i, 0]:
                        break
                sample_children[p, next_child_index[p]] = i
                next_child_index[p] += 1
    return sample_children


@njit(cache=True)
def parental_pair_markov_blankets(sample_parents, sample_children):
    n_samples = len(sample_parents)
    _, max_children = sample_children.shape
    max_blanket_size = 0
    n_pairs = 0
    pairs = {}
    for i in range(n_samples):
        p, q = sample_parents[i, 0], sample_parents[i, 1]
        # ensure ordering
        if p > q:
            p, q = q, p
        if p < 0 or q < 0:
            pass
        elif (p, q) in pairs:
            pass
        else:
            # new pair
            in_blanket = np.zeros(n_samples, dtype=np.bool_)
            in_blanket[p] = True
            in_blanket[q] = True
            for j in range(max_children):
                c = sample_children[p, j]
                if c >= 0:
                    in_blanket[c] = True
                c = sample_children[q, j]
                if c >= 0:
                    in_blanket[c] = True
            blanket = np.where(in_blanket)[0]
            max_blanket_size = max(max_blanket_size, len(blanket))
            pairs[(p, q)] = blanket
            n_pairs += 1
    parental_pairs = np.zeros((n_pairs, 2), dtype=np.int64)
    parental_pair_blankets = np.full((n_pairs, max_blanket_size), -1, dtype=np.int64)
    i = 0
    for (p, q), blanket in pairs.items():
        parental_pairs[i, 0] = p
        parental_pairs[i, 1] = q
        parental_pair_blankets[i, 0 : len(blanket)] = blanket
        i += 1
    return parental_pairs, parental_pair_blankets


@njit
def pair_allele_swap_step(
    p,
    q,
    markov_blanket,
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
    dosage,
    dosage_p,
    dosage_q,
    gamete_p,
    gamete_q,
    constraint_p,
    constraint_q,
    dosage_log_frequencies,
):
    ploidy_p = sample_ploidy[p]
    ploidy_q = sample_ploidy[q]
    index_p = np.random.randint(ploidy_p)
    index_q = np.random.randint(ploidy_q)
    allele_p = sample_genotypes[p, index_p]
    allele_q = sample_genotypes[q, index_q]
    assert allele_p >= 0
    assert allele_q >= 0
    idx = sample_read_counts[p] > 0
    read_dists_p = sample_read_dists[p][idx]
    read_counts_p = sample_read_counts[p][idx]
    read_dists_q = sample_read_dists[q][idx]
    read_counts_q = sample_read_counts[q][idx]

    # check if a new state is proposed
    if allele_p == allele_q:
        # return that no decision to make for testing etc
        return np.nan, False

    # calculate proposal ratio = g(G|G') / g(G'|G)
    proposal = count_allele(sample_genotypes[p], allele_p) * count_allele(
        sample_genotypes[q], allele_q
    )
    reversal = (1 + count_allele(sample_genotypes[p], allele_q)) * (
        1 + count_allele(sample_genotypes[q], allele_p)
    )
    lproposal_ratio = np.log(
        reversal / proposal
    )  # denominator is identical in both cases

    # current likelihood
    llk_current = 0.0
    llk_current += log_likelihood_alleles_cached(
        reads=read_dists_p,
        read_counts=read_counts_p,
        haplotypes=haplotypes,
        sample=p,
        genotype_alleles=np.sort(sample_genotypes[p, 0:ploidy_p]),
        cache=llk_cache,
    )
    llk_current += log_likelihood_alleles_cached(
        reads=read_dists_q,
        read_counts=read_counts_q,
        haplotypes=haplotypes,
        sample=q,
        genotype_alleles=np.sort(sample_genotypes[q, 0:ploidy_q]),
        cache=llk_cache,
    )

    # current prior
    lprior_current = generic_markov_blanket_log_probability(
        markov_blanket,
        sample_genotypes,
        sample_ploidy,
        sample_parents,
        gamete_tau,
        gamete_lambda,
        gamete_error,
        log_frequencies,
        dosage,
        dosage_p,
        dosage_q,
        gamete_p,
        gamete_q,
        constraint_p,
        constraint_q,
        dosage_log_frequencies,
    )

    # swap alleles
    sample_genotypes[p, index_p] = allele_q
    sample_genotypes[q, index_q] = allele_p

    # proposal likelihood
    llk_proposal = 0.0
    llk_proposal += log_likelihood_alleles_cached(
        reads=read_dists_p,
        read_counts=read_counts_p,
        haplotypes=haplotypes,
        sample=p,
        genotype_alleles=np.sort(sample_genotypes[p, 0:ploidy_p]),
        cache=llk_cache,
    )
    llk_proposal += log_likelihood_alleles_cached(
        reads=read_dists_q,
        read_counts=read_counts_q,
        haplotypes=haplotypes,
        sample=q,
        genotype_alleles=np.sort(sample_genotypes[q, 0:ploidy_q]),
        cache=llk_cache,
    )

    # proposal prior
    lprior_proposal = generic_markov_blanket_log_probability(
        markov_blanket,
        sample_genotypes,
        sample_ploidy,
        sample_parents,
        gamete_tau,
        gamete_lambda,
        gamete_error,
        log_frequencies,
        dosage,
        dosage_p,
        dosage_q,
        gamete_p,
        gamete_q,
        constraint_p,
        constraint_q,
        dosage_log_frequencies,
    )

    # calculate Metropolis-Hastings acceptance probability
    # ln(min(1, (P(G'|R)P(G')g(G|G')) / (P(G|R)P(G)g(G'|G)))
    llk_ratio = llk_proposal - llk_current
    lprior_ratio = lprior_proposal - lprior_current
    log_accept = np.minimum(0.0, llk_ratio + lprior_ratio + lproposal_ratio)
    prob_accept = np.exp(log_accept)

    # make decision
    accept = np.random.rand() < prob_accept
    if not accept:
        # put things back where you found them!
        sample_genotypes[p, index_p] = allele_p
        sample_genotypes[q, index_q] = allele_q

    # return the decision made for testing etc
    return prob_accept, accept


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
    swap_parental_alleles=True,
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
    swap_parental_alleles : bool
        If True (the default), then an allele swap step will be performed
        between each parental pairing.

    Notes
    -----
    The gamete_lambda variable only supports non-zero values when
    gametic ploidy (tau) is 2.

    Returns
    -------
    genotype_alleles_trace : ndarray, int, shape (n_samples, n_steps, ploidy)
        Genotype alleles trace for each sample.
    """
    # copy genotypes
    sample_genotypes = sample_genotypes.copy()

    # identify sample children
    sample_children = sample_children_matrix(sample_parents)

    # identify parental pairs and their Markov blankets
    parental_pairs, parental_pair_blankets = parental_pair_markov_blankets(
        sample_parents, sample_children
    )
    n_pairs = len(parental_pairs)

    # set up caches
    llk_cache = {}
    llk_cache[(-1, -1)] = np.nan

    # sample error weighting for annealing burin in
    error_weight = np.ones(n_steps, np.float64)
    if annealing:
        error_weight[0:annealing] = np.linspace(0.0, 1.0, annealing)

    # scratch arrays
    n_samples, max_ploidy = sample_genotypes.shape
    dosage = np.zeros(max_ploidy, dtype=np.int64)
    dosage_p = np.zeros(max_ploidy, dtype=np.int64)
    dosage_q = np.zeros(max_ploidy, dtype=np.int64)
    gamete_p = np.zeros(max_ploidy, dtype=np.int64)
    gamete_q = np.zeros(max_ploidy, dtype=np.int64)
    constraint_p = np.zeros(max_ploidy, dtype=np.int64)
    constraint_q = np.zeros(max_ploidy, dtype=np.int64)
    dosage_log_frequencies = np.zeros(max_ploidy, dtype=np.float64)

    # MCMC iterations
    trace = np.empty((n_steps, n_samples, max_ploidy), dtype=sample_genotypes.dtype)
    for i in range(n_steps):
        compound_step(
            sample_genotypes=sample_genotypes,
            sample_ploidy=sample_ploidy,
            sample_parents=sample_parents,
            sample_children=sample_children,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
            gamete_error=gamete_error,
            sample_read_dists=sample_read_dists,
            sample_read_counts=sample_read_counts,
            haplotypes=haplotypes,
            log_frequencies=log_frequencies,
            llk_cache=llk_cache,
            step_type=step_type,
            dosage=dosage,
            dosage_p=dosage_p,
            dosage_q=dosage_q,
            gamete_p=gamete_p,
            gamete_q=gamete_q,
            constraint_p=constraint_p,
            constraint_q=constraint_q,
            dosage_log_frequencies=dosage_log_frequencies,
        )
        if swap_parental_alleles:
            for j in range(n_pairs):
                pair_allele_swap_step(
                    p=parental_pairs[j, 0],
                    q=parental_pairs[j, 1],
                    markov_blanket=parental_pair_blankets[j],
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
                    dosage=dosage,
                    dosage_p=dosage_p,
                    dosage_q=dosage_q,
                    gamete_p=gamete_p,
                    gamete_q=gamete_q,
                    constraint_p=constraint_p,
                    constraint_q=constraint_q,
                    dosage_log_frequencies=dosage_log_frequencies,
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
