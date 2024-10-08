#!/usr/bin/env python3

import numpy as np
import numba

from mchap.jitutils import (
    random_choice,
    comb,
    structural_change,
    get_haplotype_dosage,
)
from mchap.assemble.likelihood import log_likelihood_structural_change_cached
from mchap.assemble.prior import log_genotype_prior

__all__ = [
    "interval_step",
    "compound_step",
    "random_breaks",
]


@numba.njit(cache=True)
def random_breaks(breaks, n):
    """Return a set of randomly selected non-overlapping
    intervals which cover a sequence of length n.

    Parameters
    ----------
    breaks : int
        Number of breaks between intervals (i.e. number of
        intervals - 1).
    n : int
        The combined length of intervals

    Returns
    -------
    intervals : ndarray, int shape(breaks + 1, 2)
        Ordered set of half open intervals with combined
        length of n.

    Notes
    -----
    Intervals must be greater than length zero and be
    imediately adjacent to one another with no gaps or
    overlaps.

    """

    if breaks >= n:
        raise ValueError("breaks must be smaller then n")

    indicies = np.ones(n + 1, np.bool_)
    indicies[0] = False
    indicies[-1] = False

    for _ in range(breaks):
        options = np.where(indicies)[0]
        if len(options) == 0:
            break
        else:
            point = np.random.choice(options)
            indicies[point] = False

    points = np.where(~indicies)[0]

    intervals = np.zeros((breaks + 1, 2), dtype=np.int64)

    for i in range(breaks + 1):
        intervals[i, 0] = points[i]
        intervals[i, 1] = points[i + 1]
    return intervals


@numba.njit(cache=True)
def recombination_step_n_options(labels):
    """Calculate number of unique haplotype recombination options.

    Parameters
    ----------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of some arbitrary range.

    Returns
    -------
    n : int
        Number of unique haplotype recombination options

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)

    # the dosage is used as a simple way to skip
    # compleately duplicated haplotypes
    dosage = np.empty(ploidy, np.int8)
    get_haplotype_dosage(dosage, labels)

    n = 0
    for h_0 in range(ploidy):
        if dosage[h_0] == 0:
            # this is a duplicate copy of a haplotype
            pass
        else:
            for h_1 in range(h_0 + 1, ploidy):
                if dosage[h_1] == 0:
                    # this is a duplicate copy of a haplotype
                    pass
                elif (labels[h_0, 0] == labels[h_1, 0]) or (
                    labels[h_0, 1] == labels[h_1, 1]
                ):
                    # this will result in equivilent genotypes
                    pass
                else:
                    n += 1
    return n


@numba.njit(cache=True)
def recombination_step_options(labels):
    """Calculate number of unique haplotype recombination options.

    Parameters
    ----------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of some arbitrary range.

    Returns
    -------
    option_labels : ndarray, int, shape (n, ploidy, 2)
        Labels for haplotype segments for each of n neighboring genotypes.

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)
    # the dosage is used as a simple way to skip compleately duplicated haplotypes
    dosage = np.empty(ploidy, np.int8)
    get_haplotype_dosage(dosage, labels)

    # calculate number of options
    # n_options = recombination_step_n_options(labels)
    # create options array and default to no change
    max_options = comb(ploidy, 2)
    options = np.empty((max_options, ploidy, 2), np.int8)
    for i in range(max_options):
        for j in range(ploidy):
            for k in range(2):
                options[i, j, k] = labels[j, k]

    # populate array with actual changes
    opt = 0
    for h_0 in range(ploidy):
        if dosage[h_0] == 0:
            # this is a duplicate copy of a haplotype
            pass
        else:
            for h_1 in range(h_0 + 1, ploidy):
                if dosage[h_1] == 0:
                    # this is a duplicate copy of a haplotype
                    pass
                elif (labels[h_0, 0] == labels[h_1, 0]) or (
                    labels[h_0, 1] == labels[h_1, 1]
                ):
                    # this will result in equivilent genotypes
                    pass
                else:
                    # specify recombination
                    options[opt, h_0, 0] = labels[h_1, 0]
                    options[opt, h_1, 0] = labels[h_0, 0]
                    opt += 1
    assert opt <= max_options
    return options[0:opt]


@numba.njit(cache=True)
def dosage_step_n_options(labels):
    """Calculate the number of alternative dosages within
    one steps distance.

    Parameters
    ----------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of some arbitrary range.

    Returns
    -------
    n : int
        The number of dosages within one step of the
        current dosage (excluding the current dosage).

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)

    # dosage of full haplotypes
    haplotype_dosage = np.empty(ploidy, np.int8)
    get_haplotype_dosage(haplotype_dosage, labels)

    # dosage of the segment of interest
    segment_dosage = np.empty(ploidy, np.int8)
    get_haplotype_dosage(segment_dosage, labels[:, 0:1])

    # number of options
    n = 0

    # h_0 is the potential reciever and h_1 the potential donator
    for h_0 in range(ploidy):
        if haplotype_dosage[h_0] == 0:
            # this is a full duplicate of a haplotype that has already been visited
            pass
        elif segment_dosage[h_0] == 1:
            # this would delete the only copy
            pass
        else:
            # h_0 has a segment that may be overwritten
            for h_1 in range(ploidy):
                if segment_dosage[h_1] == 0:
                    # this segment is a duplicate of another that has allready been visited
                    pass
                elif labels[h_0, 0] == labels[h_1, 0]:
                    # the haplotypes are different but the segment is identical
                    pass
                else:
                    # overwriting the segment of h_0 with the segment of h_1 will result
                    # in a novel genotype
                    n += 1
    return n


@numba.njit(cache=True)
def dosage_step_options(labels):
    """Calculate the number of alternative dosages within
    one steps distance.

    Parameters
    ----------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of some arbitrary range.

    Returns
    -------
    option_labels : ndarray, int, shape (n, ploidy, 2)
        Labels for haplotype segments for each of n neighboring genotypes.

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)

    # dosage of full haplotypes
    haplotype_dosage = np.empty(ploidy, np.int8)
    get_haplotype_dosage(haplotype_dosage, labels)

    # dosage of the segment of interest
    segment_dosage = np.empty(ploidy, np.int8)
    get_haplotype_dosage(segment_dosage, labels[:, 0:1])

    # number of options
    max_recievers = np.sum(segment_dosage[segment_dosage > 1])
    max_donors = np.sum(segment_dosage > 0) - 1
    max_options = max_recievers * max_donors

    # create options array and default to no change
    options = np.empty((max_options, ploidy, 2), np.int8)
    for i in range(max_options):
        for j in range(ploidy):
            for k in range(2):
                options[i, j, k] = labels[j, k]

    # h_0 is the potential reciever and h_1 the potential donator
    opt = 0
    for h_0 in range(ploidy):
        if haplotype_dosage[h_0] == 0:
            # this is a full duplicate of a haplotype that has already been visited
            pass
        elif segment_dosage[h_0] == 1:
            # this would delete the only copy
            pass
        else:
            # h_0 has a segment that may be overwritten
            for h_1 in range(ploidy):
                if segment_dosage[h_1] == 0:
                    # this segment is a duplicate of another that has allready been visited
                    pass
                elif labels[h_0, 0] == labels[h_1, 0]:
                    # the haplotypes are different but the segment is identical
                    pass
                else:
                    # overwriting the segment of h_0 with the segment of h_1 will result
                    # in a novel genotype
                    options[opt, h_0, 0] = labels[h_1, 0]
                    opt += 1
    assert opt <= max_options
    return options[0:opt]


@numba.njit(cache=True)
def _label_haplotypes(labels, genotype, interval=None):
    """Label each haplotype in a genotype with
    the index of its first occurance.

    Parameters
    ----------
    labels : ndarray, int, shape(ploidy, )
        An array to be updated with the haplotype labels
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as
        simple integers from 0 to n_allele.
    interval : tuple, int, optional
        If set then haplotypes are labeled using only those
        alleles withint the specified half open interval
        (defaults = None).

    Notes
    -----
    Variable `labels` is updated in place.

    See also
    --------
    haplotype_segment_labels

    """

    ploidy, n_base = genotype.shape
    labels[:] = 0

    if interval is None:
        r = range(n_base)
    else:
        r = range(interval[0], interval[1])

    for i in r:
        for j in range(1, ploidy):
            if genotype[j][i] == genotype[labels[j]][i]:
                # matches current assigned class
                pass
            else:
                # store previous assignment
                prev_label = labels[j]
                # assign to new label based on index
                # 'j' is the index and the label id
                labels[j] = j
                # check if following arrays match the new label
                for k in range(j + 1, ploidy):
                    if labels[k] == prev_label and genotype[j][i] == genotype[k][i]:
                        # this array is identical to the jth array
                        labels[k] = j


@numba.njit(cache=True)
def _interval_inverse_mask(interval, n):
    """Return a boolean vector of True values outside
    of the specified interval.

    Parameters
    ----------
    interval : tuple, int
        Half open interval.
    n : int
        Length ot the vector

    Returns:
    --------
    mask : ndarray, bool, shape (n, )
        Vector of boolean values.

    See also
    --------
    haplotype_segment_labels

    """
    if interval is None:
        mask = np.zeros(n, np.bool_)
    else:
        mask = np.ones(n, np.bool_)
        mask[interval[0] : interval[1]] = 0
    return mask


@numba.njit(cache=True)
def haplotype_segment_labels(genotype, interval=None):
    """Create a labels matrix in whihe the first coloumn contains
    labels for haplotype segments within the specified range and
    the second column contains labels for the remander of the
    haplotypes.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as
        simple integers from 0 to n_allele.
    interval : tuple, int, optional
        If set then the first label of each haplotype will
        corospond to the positions within the interval and
        the second label will corospond to the positions
        outside of the interval (defaults = None).

    Returns
    -------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of the defined interval.

    Notes
    -----
    If no interval is speciefied then the first column will contain
    labels for the full haplotypes and the second column will
    contain zeros.

    """
    ploidy, n_base = genotype.shape

    labels = np.zeros((ploidy, 2), np.int8)
    _label_haplotypes(labels[:, 0], genotype, interval=interval)
    mask = _interval_inverse_mask(interval, n_base)
    _label_haplotypes(labels[:, 1], genotype[:, mask], interval=None)
    return labels


@numba.njit(cache=True)
def interval_step(
    genotype,
    reads,
    llk,
    log_unique_haplotypes,
    inbreeding=0,
    interval=None,
    step_type=0,
    temp=1,
    read_counts=None,
    cache=None,
):
    """A structural step of an MCMC simulation consisting of
    multiple sub-steps each of which are  constrained to a single
    interval contating a sub-set of positions of a genotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        The current genotype state in an MCMC simulation consisting
        of a set of haplotypes with base positions encoded as
        integers from 0 to n_allele.
    reads : ndarray, float, shape (n_reads, n_positions, max_allele)
        Probabilistically encoded variable positions of NGS reads.
    llk : float
        The log-likelihood of the current genotype state in the MCMC
        simulation.
    log_unique_haplotypes : int
        Log of the total number of unique haplotypes possible at this locus.
    inbreeding : float
        Expected inbreeding coefficient of the genotype.
    interval : ndarray, int, shape (2, )
        The interval constraining the step.
    step_type : int
        0 for recombination or 1 for dosage swap.
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
        The log-likelihood of the genotype state after the step
        has been made.
    cache_updated : tuple
        Updated cache or None.

    Notes
    -----
    The `genotype` variable is updated in place.

    """
    assert 0 <= temp <= 1
    # labels for interval/non-interval components of current genotype
    labels = haplotype_segment_labels(genotype, interval)

    # type of structural step:
    if step_type == 0:
        option_labels = recombination_step_options(labels)
    elif step_type == 1:
        option_labels = dosage_step_options(labels)
    else:
        raise ValueError("step_type must be 0 (recombination) or 1 (dosage).")

    n_options = len(option_labels)
    if n_options == 0:
        return llk, cache
    log_proposal_prob = np.log(1 / n_options)

    # ratio of prior probabilities
    ploidy = len(genotype)
    dosage = np.empty(ploidy, dtype=np.int8)
    get_haplotype_dosage(dosage, genotype)
    lprior = log_genotype_prior(
        dosage=dosage,
        log_unique_haplotypes=log_unique_haplotypes,
        inbreeding=inbreeding,
    )

    # store values for all options and current genotype
    llks = np.empty(n_options + 1)
    llks[-1] = -np.inf
    log_accept = np.empty(n_options + 1)
    log_accept[-1] = -np.inf

    for i in range(n_options):

        # log likelihood ratio
        llk_i, cache = log_likelihood_structural_change_cached(
            reads=reads,
            genotype=genotype,
            cache=cache,
            haplotype_indices=option_labels[i, :, 0],
            interval=interval,
            read_counts=read_counts,
        )
        llks[i] = llk_i
        llk_ratio = llk_i - llk

        # calculate ratio of priors: ln(P(G')/P(G))
        get_haplotype_dosage(dosage, option_labels[i])
        lprior_i = log_genotype_prior(
            dosage=dosage,
            log_unique_haplotypes=log_unique_haplotypes,
            inbreeding=inbreeding,
        )
        lprior_ratio = lprior_i - lprior

        # balance proposal distribution
        if step_type == 0:
            n_return_options = recombination_step_n_options(option_labels[i])
        elif step_type == 1:
            n_return_options = dosage_step_n_options(option_labels[i])
        log_return_prob = np.log(1 / n_return_options)
        lproposal_ratio = log_return_prob - log_proposal_prob

        # calculate Metropolis-Hastings acceptance probability
        # ln(min(1, (P(G'|R)P(G')g(G|G')) / (P(G|R)P(G)g(G'|G)))
        mh_ratio = (llk_ratio + lprior_ratio) * temp + lproposal_ratio
        log_accept[i] = np.minimum(0.0, mh_ratio)  # max prob of log(1)

    # divide acceptance probability by number of steps to choose from
    log_accept -= np.log(n_options)

    # convert to probability of proposal * probability of acceptance
    # then fill in probability that no step is made (i.e. choose the initial state)
    probabilities = np.exp(log_accept)
    probabilities[-1] = 1 - probabilities.sum()

    # random choice of new state using probabilities
    choice = random_choice(probabilities)

    if choice < n_options:
        # update genotype
        structural_change(genotype, option_labels[choice, :, 0], interval)
        llk = llks[choice]
    else:
        # all options rejected
        pass

    # return llk of new genotype
    return llk, cache


@numba.njit(cache=True)
def compound_step(
    genotype,
    reads,
    llk,
    intervals,
    log_unique_haplotypes,
    inbreeding=0,
    step_type=0,
    randomize=True,
    temp=1,
    read_counts=None,
    cache=None,
):
    """A structural step of an MCMC simulation consisting of
    multiple sub-steps each of which are  constrained to a single
    interval contating a sub-set of positions of a genotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        The current genotype state in an MCMC simulation consisting
        of a set of haplotypes with base positions encoded as
        integers from 0 to n_allele.
    reads : ndarray, float, shape (n_reads, n_positions, max_allele)
        Probabilistically encoded variable positions of NGS reads.
    llk : float
        The log-likelihood of the current genotype state in the MCMC
        simulation.
    intervals : ndarray, int, shape (n_intervals, 2)
        The interval constraining each sub-step within this step.
    log_unique_haplotypes : float
        The log of the total number of possible haplotypes.
    inbreeding : float
        Expected inbreeding coefficient of the genotype.
    step_type : int
        0 for recombination or 1 for dosage swap.
    randomize : bool, optional
        If True then the order of substeps (as defined by the
        order of intervals) will be randomly permuted
        (default = True).
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
        The log-likelihood of the genotype state after the step
        has been made.
    cache_updated : tuple
        Updated cache or None.

    Notes
    -----
    The `genotype` variable is updated in place.

    """
    n_intervals = len(intervals)

    if randomize:
        intervals = intervals[np.random.permutation(np.arange(n_intervals))]

    # step through every interval
    for i in range(n_intervals):
        llk, cache = interval_step(
            genotype=genotype,
            reads=reads,
            llk=llk,
            cache=cache,
            log_unique_haplotypes=log_unique_haplotypes,
            inbreeding=inbreeding,
            interval=intervals[i],
            step_type=step_type,
            temp=temp,
            read_counts=read_counts,
        )
    return llk, cache
