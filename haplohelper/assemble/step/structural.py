#!/usr/bin/env python3

import numpy as np
import numba

from haplohelper.assemble.step import util



@numba.njit
def structural_change(genotype, haplotype_indices, interval=None):
    """Mutate genotype by re-arranging haplotypes within a given interval.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    haplotype_indices : array_like, int, shape (ploidy)
        Indicies of haplotypes to use within the changed interval.
    interval : tuple, int
        Interval of base-positions to swap (defaults to all base positions).
    
    Returns
    -------
    None

    Notes
    -----
    Variables `genotype` is updated in place.

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
def log_likelihood_structural_change(reads, genotype, haplotype_indices, interval=None):
    """Log likelihood of observed reads given a genotype given a structural change.

    Parameters
    ----------
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    haplotype_indices : array_like, int, shape (ploidy)
        Indicies of haplotypes to use within the changed interval.
    interval : tuple, int
        Interval of base-positions to swap (defaults to all base positions).

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype.
    """
    ploidy, n_base = genotype.shape
    n_reads = len(reads)
        
    intvl = util.interval_as_range(interval, n_base)
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                
                # check if in the altered region
                if j in intvl:
                    # use base from alternate hap
                    h_ = haplotype_indices[h]
                else:
                    # use base from current hap
                    h_ = h
                
                # get nucleotide index
                i = genotype[h_, j]

                read_hap_prod *= reads[r, j, i]
                
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk


@numba.njit
def recombination_step_n_options(labels):
    """Calculate number of unique haplotype recombination options.
    
    Parameters
    ----------
    labels : array_like, int, shape (n_haps, 2)
        Labels for haplotype sections (excluding duplicated haplotypes).

    Returns
    -------
    n : int
        Number of unique haplotype recombination options

    Notes
    -----
    Duplicate copies of haplotypes must be excluded from the labels array.

    """
    ploidy = len(labels)

    # the dosage is used as a simple way to skip compleately duplicated haplotypes
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
    labels : array_like, int, shape (ploidy, 2)
        Labels for haplotype sections.

    Returns
    -------
    options : array_like, int, shape (n_options, ploidy)
        Possible recombinations between pairs of unique haplotypes based on the dosage.

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
    """Calculate the number of alternative dosages within one steps distance.
    Dosages must include at least one copy of each unique haplotype.

    Parameters
    ----------
    labels : array_like, int, shape (ploidy, 2)
        Array of dosages.

    Returns
    -------
    n : int
        The number of dosages within one step of the current dosage (excluding the current dosage).
 

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
    """Calculate the number of alternative dosages within one steps distance.
    Dosages must include at least one copy of each unique haplotype.

    Parameters
    ----------
    labels : array_like, int, shape (ploidy, 2)
        Array of dosages.

    Returns
    -------
    n : int
        The number of dosages within one step of the current dosage (excluding the current dosage).
 

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
def interval_step(
        genotype, 
        reads, 
        llk, 
        interval=None, 
        allow_recombinations=True,
        allow_dosage_swaps=True,
        allow_deletions=False
    ):

    ploidy, _ = genotype.shape

    labels = util.haplotype_segment_labels(genotype, interval)

    # calculate number of potential steps from current state
    if allow_recombinations:
        n_recombine = recombination_step_n_options(labels)
    else:
        n_recombine = 0
    if allow_dosage_swaps:
        n_dosage = dosage_step_n_options(labels, allow_deletions)
    else:
        n_dosage = 0
    n_steps = n_recombine + n_dosage

    # not stepping is also an option
    n_options = n_steps + 1

    # array of step options
    steps = np.empty((n_steps, ploidy), np.int8)
    if allow_recombinations:
        steps[0:n_recombine] = recombination_step_options(labels)
    if allow_dosage_swaps:
        steps[n_recombine:] = dosage_step_options(labels, allow_deletions)

    # log liklihood for each new option and the current state
    llks = np.empty(n_options)
    
    # iterate through new options and calculate log-likelihood
    for opt in range(n_steps):
        llks[opt] = log_likelihood_structural_change(
            reads, 
            genotype, 
            steps[opt],
            interval
        )

    # final option is to keep the initial genotype (no recombination)
    llks[-1] = llk

    # calculate conditional probs
    conditionals = util.log_likelihoods_as_conditionals(llks)

    # choose new dosage based on conditional probabilities
    choice = util.random_choice(conditionals)
       
    if choice == (n_options - 1):
        # the choice is to keep the current state
        pass
    else:
        # update the genotype state
        structural_change(genotype, steps[choice], interval)

    # return llk of new state
    return llks[choice]


@numba.njit
def compound_step(
        genotype, 
        reads, 
        llk, 
        intervals,  
        allow_recombinations=True,
        allow_dosage_swaps=True,
        allow_deletions=False,
        randomise=True
    ):
    
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
            allow_deletions=allow_deletions
        )
    return llk
