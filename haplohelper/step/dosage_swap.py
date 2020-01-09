import numpy as np
from numba import jit

from haplohelper.step import util

@jit(nopython=True)
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



@jit(nopython=True)
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


@jit(nopython=True)
def log_likelihood_dosage(reads, genotype, dosage):
    """Log likelihood of observed reads given a genotype and a dosage of haplotypes within that genotype.

    Parameters
    ----------
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    dosage : array_like, int, shape (ploidy)
        Array of dosages.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype.

    Notes
    -----
    A haplotype with a dosage of 0 will not count towards the log-likelihood.

    """

    n_haps, n_base = genotype.shape
    n_reads = len(reads)
    
    # n_haps is not necessarily the ploidy level in this function
    # the ploidy is the sum of the dosages
    # but a dosage must be provided for each hap
    ploidy = 0
    for h in range(n_haps):
        ploidy += dosage[h]
    
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(n_haps):
            
            dose = dosage[h]
            
            if dose == 0:
                # this hap is not used (e.g. it's a copy of another)
                pass
            else:
                
                read_hap_prod = 1.0
            
                for j in range(n_base):
                    i = genotype[h, j]

                    read_hap_prod *= reads[r, j, i]
                read_prob += (read_hap_prod/ploidy) * dose
        
        llk += np.log(read_prob)
                    
    return llk


@jit(nopython=True)
def dosage_swap_step(genotype, reads, dosage, llk):
    """Dosage swap Gibbs sampler step for all haplotypes in a genotype.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    dosage : array_like, int, shape (ploidy)
        Array of initial dosages.        
    llk : float
        Log-likelihood of the initial haplotype state given the observed reads.
    
    Returns
    -------
    llk : float
        New log-likelihood of observed reads given the updated genotype dosage.

    Notes
    -----
    Variables `genotype` and `dosage` are updated in place.

    """
    # alternative dosage options
    alt_dosage_options = dosage_step_options(dosage)

    # number of options including the initial dosage
    n_options = len(alt_dosage_options) + 1
    
    # array to hold log-liklihood for each dosage option including initial
    llks = np.empty(n_options)
    
    # iterate through alternate dosage options and calculate log-likelihood
    for opt in range(0, n_options - 1):
        llks[opt] = log_likelihood_dosage(reads, genotype, alt_dosage_options[opt])

    # final option is initial dosage (no change)
    llks[-1] = llk

    # calculated denominator in log space
    log_denominator = llks[0]
    for opt in range(1, n_options):
        log_denominator = util.sum_log_prob(log_denominator, llks[opt])

    # calculate conditional probabilities
    conditionals = np.empty(n_options)
    for opt in range(n_options):
        conditionals[opt] = np.exp(llks[opt] - log_denominator)

    # ensure conditional probabilities are normalised 
    conditionals /= np.sum(conditionals)
    
    # choose new dosage based on conditional probabilities
    choice = util.random_choice(conditionals)
    
    # update dosage
    if choice == (n_options - 1):
        # this is the final option and hence the initial dosage is chosen
        pass
    else:
        # set the new dosage
        dosage[:] = alt_dosage_options[choice, :]
    
        # set the state of the haplotypes to the new dosage
        util.set_dosage(genotype, dosage)
    
    # return log-likelihood for the chosen dosage
    return llks[choice]
