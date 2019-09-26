import numpy as np
from numba import jit

from haplohelper.step import util


@jit(nopython=True)
def calculate_dosage(integer_haplotypes, dosage_array):
    """Calculates the dosage of a set of integer encoded haplotypes by 
    checking for array identity.
    
    Dosages are set in the pssed array.
    A dosage of 0 indicates that that haplotype is a duplicate of a previous haplotype
    """
    # start with assumption that all are unique
    dosage_array[:] = 1
    
    ploidy, n_base = integer_haplotypes.shape
    
    for h in range(ploidy):
        if dosage_array[h] == 0:
            # this haplotype has already been identified as equal to another
            pass
        else:
            # iterate through remaining haps
            for p in range(h+1, ploidy):
                if dosage_array[p] == 0:
                    # this haplotype has already been identified as equal to another
                    pass
                else:
                    if util.array_equal(integer_haplotypes[h], integer_haplotypes[p]):
                        dosage_array[h] += 1
                        dosage_array[p] = 0

@jit(nopython=True)
def calculate_n_dosage_swap_options(dosage_array):
    n_donors = 0
    n_recievers = 0
    ploidy = len(dosage_array)
    for h in range(ploidy):
        if dosage_array[h] == 1:
            n_recievers += 1
        elif dosage_array[h] > 1:
            n_recievers += 1
            n_donors += 1
        else:
            # 0 is an empty slot
            pass
    return 1 + (n_donors * (n_recievers - 1))


def calculate_dosage_swap_options(dosage_array, dosage_option_array):
    
    ploidy = len(dosage_array)

    # First option is keep the same dosage
    # all others are initially the same as the origional
    dosage_option_array[:] = dosage_array
    
    # start option index at 1
    option = 1
    
    for d in range(ploidy):
        if dosage_array[d] <= 1:
            # this is not a valid donor
            pass
        else:
            for r in range(ploidy):
                if r == d:
                    # can't donate to yourself
                    pass
                elif dosage_array[r] == 0:
                    # this is an empty gap
                    pass
                else:
                    # this is a valid reciever
                    # remove 1 copy from the donor and assign it to the reciever
                    dosage_option_array[option, d] -= 1
                    dosage_option_array[option, r] += 1
                    
                    # incriment to the next option
                    option += 1



@jit(nopython=True)
def set_haplotypes_to_dosage(haplotype_state, dosage_array):
    
    ploidy = len(haplotype_state)
    
    p = 0
    
    for h in range(ploidy):

        while dosage_array[h] > 1:
            
            # don't iter over dosages we know are no longer 0
            for p in range(p, ploidy):
                
                if dosage_array[p] == 0:
                    
                    haplotype_state[p] = haplotype_state[h]
                    dosage_array[h] -= 1
                    dosage_array[p] += 1



@jit(nopython=True)
def jit_log_like_dosage(reads, integer_haps, dosage_array):
    """ Assumes that sum of the dosage array is equal to the ploidy
    """

    n_haps, n_base = integer_haps.shape
    n_reads = len(reads)
    
    # n_haps is not necessarily the ploidy level in this function
    # the ploidy is the sum of the dosages
    # but a dosage must be provided for each hap
    ploidy = 0
    for h in range(n_haps):
        ploidy += dosage_array[h]
    
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(n_haps):
            
            dosage = dosage_array[h]
            
            if dosage == 0:
                # this hap is not used (e.g. it's a copy of another)
                pass
            else:
                
                read_hap_prod = 1.0
            
                for j in range(n_base):
                    i = integer_haps[h, j]

                    read_hap_prod *= reads[r, j, i]
                read_prob += (read_hap_prod/ploidy) * dosage
        
        llk += np.log(read_prob)
                    
    return llk



def dosage_swap_step(haplotype_state, dosage_array, reads):
    """haplotype_state is updated in place, log liklihood is returned.
    Assumes you have already calculated the dossage.
    """
    
    # haplotypes are encoded as integers
    ploidy, n_base = haplotype_state.shape
    
    n_dosage_swap_options = calculate_n_dosage_swap_options(dosage_array)
    options = np.arange(n_dosage_swap_options)
    
    # can we avoid creating this array dynamically by iterating options one at a time?
    # this would alow for a fixed size array
    dosage_option_array = np.empty((n_dosage_swap_options, ploidy), dtype=np.int8)
    calculate_dosage_swap_options(dosage_array, dosage_option_array)
    
    # log liklihood for each dosage option
    llks = np.empty(n_dosage_swap_options)
    
    for opt in range(n_dosage_swap_options):
        llks[opt] = jit_log_like_dosage(reads, haplotype_state, dosage_option_array[opt])

    # calculated denominator in log space
    log_denominator = llks[0]
    for opt in range(1, n_dosage_swap_options):
        log_denominator = util.sum_log_prob(log_denominator, llks[opt])

    # calculate conditional probabilities
    conditionals = np.empty(n_dosage_swap_options)
    
    for opt in range(n_dosage_swap_options):
        conditionals[opt] = np.exp(llks[opt] - log_denominator)

    # ensure conditional probabilities are normalised 
    conditionals /= np.sum(conditionals)
    
    # choose new dosage based on conditional probabilities
    choice = util.rand_choice(options, conditionals)
    
    # set the new dosage
    dosage_array = dosage_option_array[choice]
    
    # set the state of the haplotypes
    set_haplotypes_to_dosage(haplotype_state, dosage_option_array[choice])
    
    # return llk of new state
    return llks[choice]
