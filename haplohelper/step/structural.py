import numpy as np
from numba import njit

from haplohelper.step import util, recombination, dosage_swap

@njit
def interval_step(genotype, reads, llk, interval=None, allow_deletions=False):

    ploidy, _ = genotype.shape

    labels = util.haplotype_segment_labels(genotype, interval)

    # calculate number of potential steps from current state
    n_recombine = recombination.recombination_step_n_options(labels)
    n_dosage = dosage_swap.dosage_step_n_options(labels, allow_deletions)
    n_steps = n_recombine + n_dosage

    # not stepping is also an option
    n_options = n_steps + 1

    # array of step options
    steps = np.empty((n_steps, ploidy), np.int8)
    steps[0:n_recombine] = recombination.recombination_step_options(labels)
    steps[n_recombine:] = dosage_swap.dosage_step_options(labels, allow_deletions)

    # log liklihood for each new option and the current state
    llks = np.empty(n_options)
    
    # iterate through new options and calculate log-likelihood
    for opt in range(n_steps):
        llks[opt] = util.log_likelihood_structural_change(
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
        util.structural_change(genotype, steps[choice], interval)

    # return llk of new state
    return llks[choice]


@njit
def compound_step(genotype, reads, llk, intervals, randomise=True, allow_deletions=False):
    
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
            allow_deletions=allow_deletions
        )
    return llk


@njit
def compound_windowed_step(genotype, reads, llk, window_size, allow_deletions=False):

    _, n_base = genotype.shape
    
    # create intervals windowed across genotype
    intervals = util.interval_windows(window_size, n_base)
    n_intervals = len(intervals)

    # randomise interval order
    intervals = intervals[np.random.permutation(np.arange(n_intervals))]

    # step through every iterval
    for i in range(n_intervals):
        llk = interval_step(
            genotype, 
            reads, 
            llk, 
            interval=intervals[i], 
            allow_deletions=allow_deletions
        )
    return llk





    
