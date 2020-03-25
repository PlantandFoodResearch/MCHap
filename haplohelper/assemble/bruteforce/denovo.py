#!/usr/bin/env python3

import numpy as np 
import numba

from haplohelper.assemble.step import util
from haplohelper.assemble import complexity
from haplohelper.assemble.likelihood import log_likelihood


@numba.njit
def _denovo_brute_force(reads, ploidy, u_alleles, u_haps, u_gens, keep_n=0):
    
    # sanity check
    if keep_n > u_gens:
        # can't keep what doesn't exist
        keep_n = 0
    
    # complexity bounds
    _, n_base, n_nucl = reads.shape
    
    # log of number of total posible genotype permutations
    # this is used to calculate prior probability of each genotype
    log_u_perms = np.log(u_haps ** ploidy)
    
    # ready arrays for storing genotypes and associated adjusted log likelihoods
    if keep_n:
        # store best n genotypes
        genotype_array = np.empty((keep_n, ploidy, n_base), np.int8)
        genotype_llk_array = np.empty(keep_n, np.float64)
        genotype_llk_array[:] = np.nan  # initialise with nan
    else:
        # store every genotype
        genotype_array = np.empty((u_gens, ploidy, n_base), np.int8)
        genotype_llk_array = np.empty(u_gens, np.float64)

    # first genotype (zeros) is homozygous for reference
    # TODO: iterate genotypes directly rather than their indicies
    hap_indicies = np.zeros(ploidy, np.int64)
    
    # first genotype (zeros) is homozygous
    dosage = np.zeros(ploidy, np.int8)
    dosage[0] = ploidy 
    
    # first genotype (zeros)
    genotype = np.zeros((ploidy, n_base), dtype=np.int8)

    # save first genotype (zeros)
    genotype_array[0] = genotype
    
    # number of unique perms for fully homozygous first genotype is always 1
    g_perms = 1
    
    # llk for first genotype
    llk = log_likelihood(reads, genotype)
    
    # llk adjusted by prior for first genotype
    llk_adj = llk + (np.log(g_perms) - log_u_perms)
    
    # start summing all lk_adj within log space
    llk_adj_sum = llk_adj

    # iter through all other genotypes
    idx = 0 # used only if keeping all
    final = 0
    last_index = ploidy - 1
    while final < (u_haps - 1):

        searching = True 

        for i in range(ploidy):

            if searching:

                if hap_indicies[i] == final:
                    hap_indicies[i] += 1

                    # update haplotype and dosage
                    util.haplotype_of_int(genotype[i], hap_indicies[i], u_alleles)

                    searching = False

                    if i == last_index:
                        # this is the final so incriment
                        final += 1
                    else:
                        # the final will be zerod out
                        final = 0

            else:
                # zero out remaining values
                hap_indicies[i] = 0
                # update genotype
                util.haplotype_of_int(genotype[i], 0, u_alleles)

                
        # update dosage  
        util.get_dosage(dosage, hap_indicies.reshape(-1,1))
        
        # count permutations for this dosage
        g_perms = util.count_genotype_perterbations_20(dosage)
        
        # log likelihood for this genotype
        llk = log_likelihood(reads, genotype)
        
        # adjusted for prior
        llk_adj = llk + (np.log(g_perms) - log_u_perms)
        
        # add to sumtotal in log space
        llk_adj_sum = util.add_log_prob(llk_adj_sum, llk_adj)


        
        # save genotype details depending on settings
        
        if keep_n:
            # only keep if it is one of the top n most likely genotypes
            
            # index of the lowest llk value (np.nan if present, first value if duplicate)
            # TODO: simplify logic with np.argmin when fixed: https://github.com/numba/numba/pull/4210
            no_nans = True
            for i, val in enumerate(genotype_llk_array):
                if np.isnan(val):
                    no_nans = False
                    idx = i
                    break
            if no_nans:
                idx = np.argmin(genotype_llk_array)
            
            if genotype_llk_array[idx] >= llk_adj:
                # this genotype is not better than current selection
                pass
            else:
                # replace worse genotype (or nan genotype)
                genotype_array[idx] = genotype
                genotype_llk_array[idx] = llk_adj
                
        else:
            # increment genotype index
            idx += 1  
            
            # keeping all genotypes
            genotype_array[idx] = genotype
            genotype_llk_array[idx] = llk_adj

    
    # calculate posterior probabilities for saved genotypes
    probs = np.exp(genotype_llk_array - llk_adj_sum)
        
    return genotype_array, probs


class DeNovoBruteAssembler(object):
    
    def __init__(self, ploidy, keep_n=0):

        self.ploidy=ploidy
        self.keep_n=keep_n

        self.genotypes=None
        self.probabilities=None 

    def fit(self, reads):

        # calculate complexity bounds
        u_alleles = complexity.count_unique_alleles(reads)

        u_haps = complexity.count_unique_haplotypes(u_alleles)

        u_gens = complexity.count_unique_genotypes(u_haps, self.ploidy)

        # calculate genotypes and posterior probabilities
        genotypes, probs = _denovo_brute_force(
            reads, 
            ploidy=self.ploidy, 
            u_alleles=u_alleles,
            u_haps=u_haps, 
            u_gens=u_gens, 
            keep_n=self.keep_n
        )

        self.genotypes = genotypes
        self.probabilities = probs

    def posterior(self, sort=True):

        if sort:
            idx = np.flip(np.argsort(self.probabilities))

            return self.genotypes[idx], self.probabilities[idx]

        else:
            return self.genotypes.copy(), self.probabilities.copy()

