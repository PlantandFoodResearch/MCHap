import numpy as np
from numba import njit
from itertools import combinations_with_replacement

from mchap.combinatorics import count_unique_genotypes
from mchap.assemble.likelihood import log_likelihood
from mchap.calling.prior import log_genotype_prior
from mchap.jitutils import (
    increment_genotype,
    normalise_log_probs,
    genotype_alleles_as_index,
    index_as_genotype_alleles,
    add_log_prob,
)


@njit(cache=True)
def _call_posterior_mode(
    reads,
    ploidy,
    haplotypes,
    n_genotypes,
    read_counts=None,
    inbreeding=0,
    frequencies=None,
):
    """Call posterior mode genotype from a set of known haplotypes."""
    n_alleles = len(haplotypes)
    genotype = np.zeros(ploidy, np.int64)

    mode_idx = 0
    mode_llk = -np.inf
    mode_ljoint = -np.inf
    total_ljoint = -np.inf

    for i in range(n_genotypes):
        # log likelihood
        llk = log_likelihood(
            reads=reads,
            genotype=haplotypes[genotype],
            read_counts=read_counts,
        )
        # log prior
        lpr = log_genotype_prior(
            genotype, n_alleles, inbreeding=inbreeding, frequencies=frequencies
        )
        # scaled log posterior
        ljoint = llk + lpr
        if ljoint > mode_ljoint:
            # new posterior mode found
            mode_idx = i
            mode_llk = llk
            mode_ljoint = ljoint
        # normalising constant
        total_ljoint = add_log_prob(total_ljoint, ljoint)
        increment_genotype(genotype)

    mode_genotype = index_as_genotype_alleles(mode_idx, ploidy)
    return mode_genotype, mode_llk, mode_ljoint, total_ljoint


def _genotype_support_log_joint(
    genotype, reads, haplotypes, read_counts=None, inbreeding=0, frequencies=None
):
    """Calculate genotype support posterior probability from a genotype and a set of known haplotypes."""
    ploidy = len(genotype)
    # unique alleles
    support = np.unique(genotype)
    n_genotype_alleles = len(support)
    remainder = ploidy - n_genotype_alleles
    # possible dosage configurations
    options = list(combinations_with_replacement(support, remainder))

    tmp_genotype = np.zeros(ploidy, dtype=genotype.dtype)

    support_ljoint = -np.inf
    for opt in options:
        # get sorted genotype alleles
        tmp_genotype[0:n_genotype_alleles] = support
        tmp_genotype[n_genotype_alleles:ploidy] = opt
        tmp_genotype = np.sort(tmp_genotype)
        # log likelihood
        llk = log_likelihood(
            reads=reads,
            genotype=haplotypes[tmp_genotype],
            read_counts=read_counts,
        )
        # log prior
        lpr = log_genotype_prior(
            tmp_genotype,
            len(haplotypes),
            inbreeding=inbreeding,
            frequencies=frequencies,
        )
        # scaled log posterior
        ljoint = llk + lpr

        # support posterior is sum of its genotype posteriors
        support_ljoint = add_log_prob(support_ljoint, ljoint)
    return support_ljoint


@njit(cache=True)
def _posterior_allele_frequencies(
    ldenominator,
    reads,
    ploidy,
    haplotypes,
    n_genotypes,
    read_counts=None,
    inbreeding=0,
    frequencies=None,
):
    """Calculate posterior mean allele frequencies."""
    n_alleles = len(haplotypes)
    genotype = np.zeros(ploidy, np.int64)
    freqs = np.zeros(n_alleles, dtype=np.float64)
    occur = np.zeros(n_alleles, dtype=np.float64)
    for _ in range(n_genotypes):
        # log likelihood
        llk = log_likelihood(
            reads=reads,
            genotype=haplotypes[genotype],
            read_counts=read_counts,
        )
        # log prior
        lpr = log_genotype_prior(
            genotype, n_alleles, inbreeding=inbreeding, frequencies=frequencies
        )
        # scaled log posterior
        ljoint = llk + lpr
        # posterior prob
        prob = np.exp(ljoint - ldenominator)
        for i in range(ploidy):
            a = genotype[i]
            freqs[a] += prob
            # probability of occurrence
            if i == 0:
                occur[a] += prob
            else:
                # alleles are sorted ascending
                if a != genotype[i - 1]:
                    occur[a] += prob
        # next genotype
        increment_genotype(genotype)
    return freqs / ploidy, occur


def posterior_mode(
    reads,
    ploidy,
    haplotypes,
    read_counts=None,
    inbreeding=0,
    frequencies=None,
    return_support_prob=False,
    return_posterior_frequencies=False,
    return_posterior_occurrence=False,
):
    """Call posterior mode genotype with statistics from a set of known haplotypes.

    This function has a lower memory requirement than computing and storing the
    entire posterior distribution in memory.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        A set of probabalistically encoded reads.
    ploidy : int
        Ploidy of organism.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes in VCF allele order.
    read_counts : ndarray, int, shape (n_reads, )
        Counts of each (unique) read.
    inbreeding : float
        Expected inbreeding coefficient of genotype.
    frequencies : ndarray, float , shape (n_haplotypes, )
        Optional prior frequencies for each haplotype allele.
    return_support_prob : bool
        Return the mode genotype support probability (default = false).
    return_posterior_frequencies : bool
        Return posterior mean allele frequencies (default = false).
    return_posterior_occurrence : bool
        Return posterior probability of an allele occurring (default = false).

    Returns
    -------
    mode_alleles : ndarray, int, shape (ploidy, )
        The alleles (indices of input haplotypes) of the mode genotype.
    mode_llk : float
        Log likelihood of the mode genotype.
    mode_probability : float
        Posterior probability of the mode genotype.
    mode_support_probability : float
        Sum posterior probability of all genotypes within the mode genotype support.
    mean_allele_frequencies : ndarray, float, shape (n_alleles, )
        Posterior mean allele frequencies.
    allele_occurrence_probability : ndarray, float, shape (n_alleles, )
        Posterior probability of alleles occurring at any dosage.

    Notes
    -----
    This method avoids storing values for all possible genotypes in memory.
    """
    n_haplotypes = len(haplotypes)
    n_genotypes = count_unique_genotypes(n_haplotypes, ploidy)
    mode_genotype, mode_llk, mode_ljoint, total_ljoint = _call_posterior_mode(
        reads=reads,
        ploidy=ploidy,
        haplotypes=haplotypes,
        n_genotypes=n_genotypes,
        read_counts=read_counts,
        inbreeding=inbreeding,
        frequencies=frequencies,
    )
    mode_genotype_prob = np.exp(mode_ljoint - total_ljoint)

    result = [mode_genotype, mode_llk, mode_genotype_prob]

    if return_support_prob:
        support_ljoint = _genotype_support_log_joint(
            genotype=mode_genotype,
            reads=reads,
            haplotypes=haplotypes,
            read_counts=read_counts,
            inbreeding=inbreeding,
            frequencies=frequencies,
        )
        mode_support_prob = np.exp(support_ljoint - total_ljoint)
        result.append(mode_support_prob)

    if return_posterior_frequencies or return_posterior_occurrence:
        mean_frequencies, occurrence = _posterior_allele_frequencies(
            ldenominator=total_ljoint,
            reads=reads,
            ploidy=ploidy,
            haplotypes=haplotypes,
            n_genotypes=n_genotypes,
            read_counts=read_counts,
            inbreeding=inbreeding,
            frequencies=frequencies,
        )
        if return_posterior_frequencies:
            result.append(mean_frequencies)
        if return_posterior_occurrence:
            result.append(occurrence)

    return tuple(result)


@njit(cache=True)
def _genotype_likelihoods(reads, ploidy, haplotypes, n_genotypes, read_counts=None):
    likelihoods = np.full(n_genotypes, np.nan, np.float32)
    genotype = np.zeros(ploidy, np.int64)
    for i in range(0, n_genotypes):
        likelihoods[i] = log_likelihood(
            reads=reads,
            genotype=haplotypes[genotype],
            read_counts=read_counts,
        )
        increment_genotype(genotype)
    return likelihoods


def genotype_likelihoods(reads, ploidy, haplotypes, read_counts=None):
    """Calculate the log likelihood of every possible genotype
    for a given set of reads, ploidy, and possible haplotypes.

    Parameters
    ----------
    reads : ndarray, float, shape (n_reads, n_pos, n_nucl)
        A set of probabalistically encoded reads.
    ploidy : int
        Ploidy of organism.
    haplotypes : ndarray, int, shape (n_haplotypes, n_pos)
        Integer encoded haplotypes in VCF allele order.

    Returns
    -------
    log_likelihoods : ndarray, float, shape (n_genotypes, )
        VCF ordered genotype log likelihoods.
    """
    n_haplotypes = len(haplotypes)
    n_genotypes = count_unique_genotypes(n_haplotypes, ploidy)
    return _genotype_likelihoods(
        reads=reads,
        ploidy=ploidy,
        haplotypes=haplotypes,
        n_genotypes=n_genotypes,
        read_counts=read_counts,
    )


@njit(cache=True)
def genotype_posteriors(
    log_likelihoods, ploidy, n_alleles, inbreeding=0, frequencies=None
):
    """Calculate posterior probability of every possible genotype
    for a given set of likelihoods, ploidy, and number of alleles.

    Parameters
    ----------
    log_likelihoods : ndarray, float, shape(n_genotypes, )
        VCF ordered log (natural) likelihood of each possible genotype.
    ploidy : int
        Ploidy of organism.
    n_alleles : int
        Total number of possible (haplotype) alleles at this locus.
    inbreeding : float
        Inbreeding coefficient of organism used when calculating prior
        probabilities.
    frequencies : ndarray, float , shape (n_haplotypes, )
        Optional prior frequencies for each haplotype allele.

    Returns
    -------
    posteriors : ndarray, float, shape(n_genotypes, )
        VCF ordered posterior probability of each possible genotype.
    """
    n_genotypes = len(log_likelihoods)
    posteriors = np.zeros(n_genotypes, dtype=log_likelihoods.dtype)
    genotype = np.zeros(ploidy, np.int64)
    for i in range(n_genotypes):
        llk = log_likelihoods[i]
        lpr = log_genotype_prior(
            genotype, n_alleles, inbreeding=inbreeding, frequencies=frequencies
        )
        posteriors[i] = llk + lpr
        increment_genotype(genotype)
    return normalise_log_probs(posteriors)


@njit(cache=True)
def posterior_allele_frequencies(posteriors, ploidy, n_alleles):
    """Calculate posterior mean allele frequencies of every allele
    for a given posteriors distribution.

    Parameters
    ----------
    posteriors : ndarray, float, shape(n_genotypes, )
        VCF ordered posterior probabilities.
    ploidy : int
        Ploidy of organism.
    n_alleles : int
        Total number of possible (haplotype) alleles at this locus.

    Returns
    -------
    mean_allele_frequencies : ndarray, float, shape (n_alleles, )
        Posterior mean allele frequencies.
    posterior_allele_counts : ndarray, float, shape (n_alleles, )
        Posterior allele counts
    allele_occurrence_probability : ndarray, float, shape (n_alleles, )
        Posterior probability of alleles occurring at any dosage.
    """
    n_genotypes = len(posteriors)
    counts = np.zeros(n_alleles, dtype=np.float64)
    occur = np.zeros(n_alleles, dtype=np.float64)
    genotype = np.zeros(ploidy, np.int64)
    for i in range(n_genotypes):
        p = posteriors[i]
        for j in range(ploidy):
            a = genotype[j]
            counts[a] += p
            if j == 0:
                occur[a] += p
            elif a != genotype[j - 1]:
                occur[a] += p
        increment_genotype(genotype)
    return counts / ploidy, counts, occur


def alternate_dosage_posteriors(genotype_alleles, probabilities):
    """Extract alternate dosage probabilities based on genotype support.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Integers indicating alleles.
    probabilities : ndarray, float, shape (n_genotypes, )
        Posterior probability of every possible genotype in VCF order.

    Returns
    -------
    support_posteriors : ndarray, float, shape (n_alternate_dosages, )
        Posterior probability of each alternate dosage in order.

    """
    ploidy = len(genotype_alleles)
    support = np.unique(genotype_alleles)
    n_alleles = len(support)
    array = np.zeros(ploidy, dtype=genotype_alleles.dtype)
    remainder = ploidy - n_alleles
    options = list(combinations_with_replacement(support, remainder))
    n_options = len(options)
    probs = np.zeros(n_options, float)
    indices = np.zeros(n_options, int)
    genotypes = np.zeros((n_options, ploidy), int)
    for i, opt in enumerate(options):
        array[0:n_alleles] = support
        array[n_alleles:ploidy] = opt
        array = np.sort(array)
        genotypes[i] = array.copy()
        idx = genotype_alleles_as_index(array)
        indices[i] = idx
        probs[i] = probabilities[idx]
    idx = np.argsort(indices)
    return genotypes[idx], probs[idx]
