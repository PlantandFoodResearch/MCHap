import numpy as np
import scipy
from numba import jit

from haplohelper.step import util



def recombination_point_probabilities(n_base, a=1, b=1):
    dist = scipy.stats.beta(a, b)
    points = np.arange(1, n_base ) / (n_base - 1)
    probs = dist.cdf(points)
    probs[1:] = probs[1:] - probs[:-1]
    return probs


@jit(nopython=True)
def calculate_n_unique_recombination_options(dosage_array):
    ploidy = len(dosage_array)
    n_unique_haps = 0
    for h in range(ploidy):
        if dosage_array[h] > 0:
            n_unique_haps += 1
    return ((n_unique_haps -1) * n_unique_haps) // 2 + 1  # + 1 for no change


@jit(nopython=True)
def calculate_unique_recombination_options(dosage_array, recombination_option_array):
    ploidy = len(dosage_array)
    
    # first option is no change
    # TODO: the first option shouldn't need to be recalculated
    recombination_option_array[0, :] = 0
    opt = 1
    for h_0 in range(ploidy):
        if dosage_array[h_0] < 1:
            pass
        else:
            for h_1 in range(h_0 + 1, ploidy):
                if dosage_array[h_1] < 1:
                    pass
                else:
                    recombination_option_array[opt, 0] = h_0
                    recombination_option_array[opt, 1] = h_1
                    opt+=1


@jit(nopython=True)
def jit_log_like_recombination(reads, integer_haps, h_x, h_y, point):
    
    ploidy, n_base = integer_haps.shape
    n_reads = len(reads)
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                
                if h == h_x and j >= point:
                    i = integer_haps[h_y, j]
                elif h == h_y and j >= point:
                    i = integer_haps[h_x, j]
                else:
                    i = integer_haps[h, j]

                read_hap_prod *= reads[r, j, i]
                
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk


def recombination_step(haplotype_state, reads, dosage_array, recombination_point):
    """haplotype_state is updated in place, log liklihood is returned.
    Assumes you have already calculated the dossage.
    """
    
    # haplotypes are encoded as integers
    ploidy, n_base = haplotype_state.shape
    
    # set the current dosage
    util.set_dosage_to_genotype(haplotype_state, dosage_array)
    
    n_unique_recombination_options = calculate_n_unique_recombination_options(dosage_array)
    options = np.arange(n_unique_recombination_options)
    
    # can we avoid creating this array dynamically by iterating options one at a time?
    # this would alow for a fixed size array
    recombination_option_array = np.zeros((n_unique_recombination_options, 2), dtype=np.int8)
    calculate_unique_recombination_options(dosage_array, recombination_option_array)
    
    # log liklihood for each recombination option
    llks = np.empty(n_unique_recombination_options)
    
    for opt in range(n_unique_recombination_options):
        llks[opt] = jit_log_like_recombination(
            reads, 
            haplotype_state, 
            recombination_option_array[opt, 0],
            recombination_option_array[opt, 1],
            recombination_point
        )

    # calculated denominator in log space
    log_denominator = llks[0]
    for opt in range(1, n_unique_recombination_options):
        log_denominator = util.sum_log_prob(log_denominator, llks[opt])

    # calculate conditional probabilities
    conditionals = np.empty(n_unique_recombination_options)
    
    for opt in range(n_unique_recombination_options):
        conditionals[opt] = np.exp(llks[opt] - log_denominator)

    # ensure conditional probabilities are normalised 
    conditionals /= np.sum(conditionals)
    
    # choose new dosage based on conditional probabilities
    choice = util.random_choice(options, conditionals)
       
    if choice == 0:
        # the state is not changed
        pass
    else:
        # set the state of the recombinated haplotypes
        h_x = recombination_option_array[choice, 0]
        h_y = recombination_option_array[choice, 1]
        
        # swap bases from the recombination point
        for j in range(recombination_point, n_base):
            
            j_x = haplotype_state[h_x, j].copy()
            j_y = haplotype_state[h_y, j].copy()
            
            haplotype_state[h_x, j] = j_y
            haplotype_state[h_y, j] = j_x
                
    # set the new dosage
    util.set_dosage_to_genotype(haplotype_state, dosage_array)

    # return llk of new state
    return llks[choice]