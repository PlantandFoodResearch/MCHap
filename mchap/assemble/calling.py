import numpy as np
from numba import njit
from itertools import combinations_with_replacement

from mchap.combinatorics import count_unique_genotypes
from mchap.assemble.likelihood import log_likelihood, log_genotype_prior
from mchap.assemble.utils import get_dosage
from mchap.jitutils import (
    increment_genotype,
    normalise_log_probs,
    genotype_alleles_as_index,
    index_as_genotype_alleles,
    add_log_prob,
)


__all__ = [
    "genotype_likelihoods",
    "genotype_posteriors",
    "call_posterior_haplotypes",
    "alternate_dosage_posteriors",
    "call_posterior_mode",
]


@njit(cache=True)
def _call_posterior_mode(
    reads, ploidy, haplotypes, n_genotypes, read_counts=None, inbreeding=0
):
    """Call posterior mode genotype from a set of known haplotypes."""
    n_alleles = len(haplotypes)
    genotype = np.zeros(ploidy, np.int64)
    dosage = np.zeros(ploidy, np.int8)

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
        get_dosage(dosage, genotype.reshape(ploidy, 1))
        lpr = log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
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


def _phenotype_log_joint(genotype, reads, haplotypes, read_counts=None, inbreeding=0):
    """Calculate phenotype posterior probability from a genotype and a set of known haplotypes."""
    ploidy = len(genotype)
    # unique alleles
    phenotype = np.unique(genotype)
    n_genotype_alleles = len(phenotype)
    remainder = ploidy - n_genotype_alleles
    # possible dosage configurations
    options = list(combinations_with_replacement(phenotype, remainder))

    array = np.zeros(ploidy, dtype=genotype.dtype)
    dosage = np.zeros(ploidy, np.int8)

    phenotype_ljoint = -np.inf
    for opt in options:
        # get sorted genotype alleles
        array[0:n_genotype_alleles] = phenotype
        array[n_genotype_alleles:ploidy] = opt
        array = np.sort(array)
        # log likelihood
        llk = log_likelihood(
            reads=reads,
            genotype=haplotypes[array],
            read_counts=read_counts,
        )
        # log prior
        get_dosage(dosage, array.reshape(ploidy, 1))
        lpr = log_genotype_prior(dosage, len(haplotypes), inbreeding=inbreeding)
        # scaled log posterior
        ljoint = llk + lpr

        # phenotype posterior is sum of its genotype posteriors
        phenotype_ljoint = add_log_prob(phenotype_ljoint, ljoint)
    return phenotype_ljoint


def call_posterior_mode(
    reads,
    ploidy,
    haplotypes,
    read_counts=None,
    inbreeding=0,
    return_phenotype_prob=True,
):
    """Call posterior mode genotype with statistics from a set of known haplotypes.

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
    return_phenotype_prob : bool
        Return the mode_phenotype_probability (default = true).

    Returns
    -------
    mode_alleles : ndarray, int, shape (ploidy, )
        The alleles (indices of input haplotypes) of the mode genotype.
    mode_llk : float
        Log likelihood of the mode genotype.
    mode_probability : float
        Posterior probability of the mode genotype.
    mode_phenotype_probability : float
        Sum posterior probability of all genotypes within the mode phenotype.

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
    )
    mode_genotype_prob = np.exp(mode_ljoint - total_ljoint)

    if not return_phenotype_prob:
        return mode_genotype, mode_llk, mode_genotype_prob

    phenotype_ljoint = _phenotype_log_joint(
        genotype=mode_genotype,
        reads=reads,
        haplotypes=haplotypes,
        read_counts=read_counts,
        inbreeding=inbreeding,
    )
    mode_phenotype_prob = np.exp(phenotype_ljoint - total_ljoint)
    return mode_genotype, mode_llk, mode_genotype_prob, mode_phenotype_prob


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
def genotype_posteriors(log_likelihoods, ploidy, n_alleles, inbreeding=0):
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

    Returns
    -------
    posteriors : ndarray, float, shape(n_genotypes, )
        VCF ordered posterior probability of each possible genotype.
    """
    n_genotypes = len(log_likelihoods)
    posteriors = np.zeros(n_genotypes, dtype=log_likelihoods.dtype)
    genotype = np.zeros(ploidy, np.int64)
    dosage = np.zeros(ploidy, np.int8)
    for i in range(n_genotypes):
        get_dosage(dosage, genotype.reshape(ploidy, 1))
        llk = log_likelihoods[i]
        lpr = log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
        posteriors[i] = llk + lpr
        increment_genotype(genotype)
    return normalise_log_probs(posteriors)


def alternate_dosage_posteriors(genotype_alleles, probabilities):
    """Extract alternate dosage probabilities based on allelic phenotype.

    Parameters
    ----------
    genotype_alleles : ndarray, int, shape (ploidy, )
        Integers indicating alleles.
    probabilities : ndarray, float, shape (n_genotypes, )
        Posterior probability of every possible genotype in VCF order.

    Returns
    -------
    phenotype_posteriors : ndarray, float, shape (n_alternate_dosages, )
        Posterior probability of each alternate dosage in order.

    """
    ploidy = len(genotype_alleles)
    phenotype = np.unique(genotype_alleles)
    n_alleles = len(phenotype)
    array = np.zeros(ploidy, dtype=genotype_alleles.dtype)
    remainder = ploidy - n_alleles
    options = list(combinations_with_replacement(phenotype, remainder))
    n_options = len(options)
    probs = np.zeros(n_options, float)
    indices = np.zeros(n_options, int)
    genotypes = np.zeros((n_options, ploidy), int)
    for i, opt in enumerate(options):
        array[0:n_alleles] = phenotype
        array[n_alleles:ploidy] = opt
        array = np.sort(array)
        genotypes[i] = array.copy()
        idx = genotype_alleles_as_index(array)
        indices[i] = idx
        probs[i] = probabilities[idx]
    idx = np.argsort(indices)
    return genotypes[idx], probs[idx]


def call_posterior_haplotypes(posteriors, threshold=0.01):
    """Call haplotype alleles for VCF output from a population
    of genotype posterior distributions.

    Parameters
    ----------
    posteriors : list, PosteriorGenotypeDistribution
        A list of individual genotype posteriors.
    threshold : float
        Minimum required posterior probability of occurrence
        with in any individual for a haplotype to be included.

    Returns
    -------
    haplotypes : ndarray, int, shape, (n_haplotypes, n_base)
        VCF sorted haplotype arrays.
    """
    # maps of bytes to arrays and bytes to sum probs
    haplotype_arrays = {}
    haplotype_values = {}
    # iterate through genotype posterors
    for post in posteriors:
        # include haps based on probability of occurrence
        haps, probs, weights = post.haplotype_probabilities(return_weighted=True)
        idx = probs >= threshold
        # order haps based on weighted prob
        haps = haps[idx]
        weights = weights[idx]
        for h, w in zip(haps, weights):
            b = h.tobytes()
            if b not in haplotype_arrays:
                haplotype_arrays[b] = h
                haplotype_values[b] = 0
            haplotype_values[b] += w
    # remove reference allele if present
    refbytes = None
    for b, h in haplotype_arrays.items():
        if np.all(h == 0):
            # ref allele
            refbytes = b
    if refbytes is not None:
        haplotype_arrays.pop(refbytes)
        haplotype_values.pop(refbytes)
    # combine all called haplotypes into array
    n_alleles = len(haplotype_arrays) + 1
    n_base = posteriors[0].genotypes.shape[-1]
    haplotypes = np.full((n_alleles, n_base), -1, np.int8)
    values = np.full(n_alleles, -1, float)
    for i, (b, h) in enumerate(haplotype_arrays.items()):
        p = haplotype_values[b]
        haplotypes[i] = h
        values[i] = p
    haplotypes[-1][:] = 0  # ref allele
    values[-1] = values.max() + 1
    order = np.flip(np.argsort(values))
    return haplotypes[order]
