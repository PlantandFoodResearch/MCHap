#!/usr/bin/env python3

import numpy as np 
import numba

from haplohelper.assemble.step import util
from haplohelper.assemble import complexity
from haplohelper.assemble.likelihood import log_likelihood


@numba.njit
def _reference_brute_force(reads, genotypes, priors):
    assert reads.shape[-2] == genotypes.shape[-1]
    
    n_gens = len(genotypes)
    assert n_gens == len(priors)
    
    # for llk adjusted by prior
    llk_adj_array = np.zeros(n_gens)
    
    # first g manually to start sum_llk_adj
    i = 0
    llk_adj = log_likelihood(reads, genotypes[i])
    llk_adj += np.log(priors[i])
    llk_adj_array[i] = llk_adj
    
    # can't start with log(0)
    sum_llk_adj = llk_adj
    
    for i in range(1, n_gens):
        llk_adj = log_likelihood(reads, genotypes[i])
        llk_adj += np.log(priors[i])
        llk_adj_array[i] = llk_adj
        
        # add to running total
        sum_llk_adj = util.add_log_prob(sum_llk_adj, llk_adj)
    
    probs = np.exp(llk_adj_array - sum_llk_adj)
    
    return probs


class GenotypeBruteCaller(object):
    
    def __init__(self, genotypes, priors):
        self.genotypes = genotypes
        self.priors = priors
        self.probabilities=None
        
    def fit(self, reads):
        
        self.probabilities = _reference_brute_force(reads, self.genotypes, self.priors)
        
    def posterior(self, sort=True):

        if sort:
            idx = np.flip(np.argsort(self.probabilities))

            return self.genotypes[idx], self.probabilities[idx]

        else:
            return self.genotypes.copy(), self.probabilities.copy()
