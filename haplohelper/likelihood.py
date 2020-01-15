#!/usr/bin/env python3

import numpy as np
import numba


@numba.njit
def log_likelihood(reads, genotype):
    """Log likelihood of observed reads given a genotype.

    Parameters
    ----------
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype.
    """
    
    ploidy, n_base = genotype.shape
    n_reads = len(reads)
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                i = genotype[h, j]

                read_hap_prod *= reads[r, j, i]
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk


@numba.njit
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
