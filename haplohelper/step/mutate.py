import numpy as np 
from numba import jit

from haplohelper.step import util

@jit(nopython=True)
def jit_log_like(reads, integer_haps):
    
    ploidy, n_base = integer_haps.shape
    n_reads = len(reads)
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                i = integer_haps[h, j]

                read_hap_prod *= reads[r, j, i]
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk


@jit(nopython=True)
def mutate_step(haplotype_state, reads):
    """Iterates through every allele of every haplotype"""
    ploidy = len(haplotype_state)
    n_reads, n_base, n_nucl = reads.shape
    
    alleles = np.arange(0, n_nucl)
    
    # cache of log likelihoods calculated with each allele
    llks = np.empty(n_nucl)
    conditionals = np.empty(n_nucl)
    
    for h in np.random.permutation(np.arange(0, ploidy)):

        for j in np.random.permutation(np.arange(0, n_base)):

            for i in range(n_nucl):
                haplotype_state[h, j] = i
                llks[i] = jit_log_like(reads, haplotype_state)

            # calculated denominator in log space
            log_denominator = llks[0]
            for i in range(1, n_nucl):
                log_denominator = util.sum_log_prob(log_denominator, llks[i])

            # calculate conditional probabilities
            for i in range(n_nucl):
                conditionals[i] = np.exp(llks[i] - log_denominator)

            # ensure conditional probabilities are normalised 
            conditionals /= np.sum(conditionals)

            # if a prior is used then it can be multiplied by probs here
            choice = util.rand_choice(alleles, conditionals)

            # update state
            haplotype_state[h, j] = choice
    
    # return final log liklihood
    return llks[choice]
    
