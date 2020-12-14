#!/usr/bin/env python3

import numpy as np
import numba

from mchap.assemble import util
from mchap.assemble.likelihood import log_likelihood_structural_change


@numba.njit
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
        raise ValueError('breaks must be smaller then n')
    
    indicies = np.ones(n+1, np.bool8)
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
    
    intervals = np.zeros((breaks + 1, 2), dtype = np.int64)
    
    for i in range(breaks + 1):
        intervals[i, 0] = points[i]
        intervals[i, 1] = points[i + 1]
    return intervals


@numba.njit
def structural_change(genotype, haplotype_indices, interval=None):
    """Mutate genotype by re-arranging haplotypes 
    within a given interval.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as 
        simple integers from 0 to n_allele.
    haplotype_indices : ndarray, int, shape (ploidy)
        Indicies of haplotypes to update alleles from.
    interval : tuple, int, optional
        If set then base-positions copies/swaps between
        haplotype is constrained to the specified 
        half open interval (defaults = None).
    
    Returns
    -------
    None

    Notes
    -----
    Variable `genotype` is updated in place.

    """
    
    ploidy, n_base = genotype.shape
        
    cache = np.empty(ploidy, dtype=np.int8)

    r = util.interval_as_range(interval, n_base)
    
    for j in r:
        
        # copy to cache 
        for h in range(ploidy):
            cache[h] = genotype[h, j]
        
        # copy new bases back to genotype
        for h in range(ploidy):
            genotype[h, j] = cache[haplotype_indices[h]]


@numba.njit
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

    Notes
    -----
    Duplicate copies of haplotypes must be excluded 
    from the labels array.

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)

    # the dosage is used as a simple way to skip 
    # compleately duplicated haplotypes
    dosage = np.empty(ploidy, np.int8)
    util.get_dosage(dosage, labels)

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
                elif (labels[h_0, 0] == labels[h_1, 0]) or (labels[h_0, 1] == labels[h_1, 1]):
                    # this will result in equivilent genotypes
                    pass
                else:
                    n += 1
    return n


@numba.njit
def recombination_step_options(labels):
    """Possible recombinations between pairs of unique haplotypes.

    Parameters
    ----------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of some arbitrary range.

    Returns
    -------
    options : ndarray, int, shape (n_options, ploidy)
        Possible recombinations between pairs of unique 
        haplotypes based on the dosage.

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)
    # the dosage is used as a simple way to skip compleately duplicated haplotypes
    dosage = np.empty(ploidy, np.int8)
    util.get_dosage(dosage, labels)
    
    # calculate number of options
    n_options = recombination_step_n_options(labels)
    # create options array and default to no change
    options = np.empty((n_options, ploidy), np.int8)
    for i in range(n_options):
        for j in range(ploidy):
            options[i, j] = j
    
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
                elif (labels[h_0, 0] == labels[h_1, 0]) or (labels[h_0, 1] == labels[h_1, 1]):
                    # this will result in equivilent genotypes
                    pass
                else:
                    # specify recombination
                    options[opt, h_0] = h_1
                    options[opt, h_1] = h_0
                    opt+=1

    return options


@numba.njit
def dosage_step_n_options(labels, allow_deletions=False):
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
 
    Notes
    -----
    Dosages must include at least one copy of each unique haplotype.

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)

    # dosage of full haplotypes
    haplotype_dosage = np.empty(ploidy, np.int8)
    util.get_dosage(haplotype_dosage, labels)

    # dosage of the segment of interest
    segment_dosage = np.empty(ploidy, np.int8)
    util.get_dosage(segment_dosage, labels[:,0:1])

    # number of options
    n = 0

    # h_0 is the potential reciever and h_1 the potential donator
    for h_0 in range(ploidy):
        if haplotype_dosage[h_0] == 0:
            # this is a full duplicate of a haplotype that has already been visited
            pass
        elif (not allow_deletions) and (segment_dosage[h_0] == 1):
            # if segment dosage is 1 then it is the only copy of this segment
            # hence overwriting it will delete the only copy of this segment
            # NOTE: if segment dosage is 0 then it is a duplicate so there must be
            # more than one copy and hence it is ok to continue and overwrite
            # likewise if segment dosage is > 1 there is more than one copy
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


@numba.njit
def dosage_step_options(labels, allow_deletions=False):
    """Calculate the number of alternative dosages within 
    one steps distance.
    Dosages must include at least one copy of each unique 
    haplotype.

    Parameters
    ----------
    labels : ndarray, int, shape (ploidy, 2)
        Labels for haplotype segments within and outside
        of some arbitrary range.
    allow_deletions : bool, optional
        Set to False to dis-allow dosage changes 
        that remove part or all of the only 
        copy of a haplotype from the genotype
        (default = True).

    Returns
    -------
    n : int
        The number of dosages within one step of the current 
        dosage (excluding the current dosage).

    See also
    --------
    haplotype_segment_labels

    """
    ploidy = len(labels)

    # dosage of full haplotypes
    haplotype_dosage = np.empty(ploidy, np.int8)
    util.get_dosage(haplotype_dosage, labels)

    # dosage of the segment of interest
    segment_dosage = np.empty(ploidy, np.int8)
    util.get_dosage(segment_dosage, labels[:,0:1])

    # number of options
    n_options = dosage_step_n_options(labels, allow_deletions=allow_deletions)
    # create options array and default to no change
    options = np.empty((n_options, ploidy), np.int8)
    for i in range(n_options):
        for j in range(ploidy):
            options[i, j] = j

    # h_0 is the potential reciever and h_1 the potential donator
    opt = 0
    for h_0 in range(ploidy):
        if haplotype_dosage[h_0] == 0:
            # this is a full duplicate of a haplotype that has already been visited
            pass
        elif (not allow_deletions) and (segment_dosage[h_0] == 1):
            # if segment dosage is 1 then it is the only copy of this segment
            # hence overwriting it will delete the only copy of this segment
            # NOTE: if segment dosage is 0 then it is a duplicate so there must be
            # more than one copy and hence it is ok to continue and overwrite
            # likewise if segment dosage is > 1 there is more than one copy
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
                    options[opt, h_0] = h_1
                    opt+=1
    return options


@numba.njit
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

    r = util.interval_as_range(interval, n_base)

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


@numba.njit
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
        mask = np.zeros(n, np.bool8)
    else:
        mask = np.ones(n, np.bool8)
        mask[interval[0]:interval[1]] = 0
    return mask


@numba.njit
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


@numba.njit
def interval_step(
        genotype, 
        reads, 
        llk, 
        interval=None, 
        allow_recombinations=True,
        allow_dosage_swaps=True,
        temp=1,
    ):
    """A structural sub-step of an MCMC simulation constrained to 
    a single interval contating a sub-set of positions of a genotype.

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
    interval : tuple, int, optional
        The interval constraining the domain of this sub-step.
    allow_recombinations : bool, optional
        Set to False to dis-allow structural steps involving
        the recombination of part of a pair of haplotypes
        (default = True).
    allow_dosage_swaps : bool, optional
        Set to False to dis-allow structural steps involving
        dosage changes between parts of a pair of haplotypes
        (default = True).
    temp : float
        An inverse temperature in the interval 0, 1 to adjust
        the sampled distribution by.

    Returns
    -------
    llk : float
        The log-likelihood of the genotype state after the step 
        has been made.

    Notes
    -----
    The `genotype` variable is updated in place.

    """
    assert  0 <= temp <= 1
    # labels for interval/non-interval components of current genotype
    labels = haplotype_segment_labels(genotype, interval)

    # calculate number of potential steps from current state
    if allow_recombinations and allow_dosage_swaps:
        steps = np.append(
            recombination_step_options(labels),
            dosage_step_options(labels),
            axis=0,
        )
    elif allow_recombinations:
        steps = recombination_step_options(labels)
    elif allow_dosage_swaps:
        steps = dosage_step_options(labels)
    else:
        raise ValueError('Must allow recombination and/or dosage steps.')

    # number of neighbouring genotypes
    n_steps = len(steps)

    # non-transition is also an option
    n_options = n_steps + 1

    # labels for interval/non-interval components of neighbouring genotypes
    neighbour_labels = labels.copy()

    # Log of ratios of proposal probabilities of transitioning to and from
    # each neighbouring genotype (required to balance proposal distribution)
    log_proposal_ratios = np.empty(n_options)
    for neighbour in range(n_steps):

        # probability of proposing a transition to neighbour
        to_neighbour_prob = 1 / n_options

        # alter within interval labels to match neighbours genotype 
        neighbour_labels[:,0] = labels[:,0][steps[neighbour]]

        # probability of proposing a transition back from neighbour
        neighbour_n_options = 1  # non-transition
        if allow_recombinations:
            neighbour_n_options += recombination_step_n_options(neighbour_labels)
        if allow_dosage_swaps:
            neighbour_n_options += dosage_step_n_options(neighbour_labels)
        from_neighbour_prob = 1 / neighbour_n_options

        # store log of ratio of probs
        ratio = from_neighbour_prob / to_neighbour_prob
        log_proposal_ratios[neighbour] = np.log(ratio)

    # final option is to keep the current genotype 
    log_proposal_ratios[-1] = np.log(1.0)

    # log liklihood for each neighbouring genotype and the current
    llks = np.empty(n_options)
    
    # iterate through neighbouring genotypes and calculate log-likelihood
    for neighbour in range(n_steps):
        llks[neighbour] = log_likelihood_structural_change(
            reads, 
            genotype, 
            steps[neighbour],
            interval
        )

    # final option is to keep the current genotype 
    llks[-1] = llk

    # log acceptance ratio for each proposed step
    log_accept = llks - llk

    # raise to the temperature
    log_accept *= temp

    # modify by proposal ratio
    log_accept += log_proposal_ratios

    # calculate conditional probs of acceptance for all transitions
    conditionals = util.log_likelihoods_as_conditionals(log_accept)

    # choose new genotype based on conditional probabilities
    choice = util.random_choice(conditionals)
       
    if choice == (n_options - 1):
        # the choice is to keep the current genotype
        pass
    else:
        # update the genotype state
        structural_change(genotype, steps[choice], interval)
        llk = llks[choice]

    # return llk of new genotype
    return llk


@numba.njit
def compound_step(
        genotype, 
        reads, 
        llk, 
        intervals,  
        allow_recombinations=True,
        allow_dosage_swaps=True,
        randomise=True,
        temp=1,
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
    allow_recombinations : bool, optional
        Set to False to dis-allow structural steps involving
        the recombination of part of a pair of haplotypes
        (default = True).
    allow_dosage_swaps : bool, optional
        Set to False to dis-allow structural steps involving
        dosage changes between parts of a pair of haplotypes
        (default = True).
    randomise : bool, optional
        If True then the order of substeps (as defined by the 
        order of intervals) will be randomly permuted
        (default = True).
    temp : float
        An inverse temperature in the interval 0, 1 to adjust
        the sampled distribution by.

    Returns
    -------
    llk : float
        The log-likelihood of the genotype state after the step 
        has been made.

    Notes
    -----
    The `genotype` variable is updated in place.

    """
    
    n_intervals = len(intervals)

    if randomise:
        intervals = intervals[np.random.permutation(np.arange(n_intervals))]

    # step through every iterval
    for i in range(n_intervals):
        llk = interval_step(
            genotype, 
            reads, 
            llk, 
            interval=intervals[i], 
            allow_recombinations=allow_recombinations,
            allow_dosage_swaps=allow_dosage_swaps,
            temp=temp,
        )
    return llk
