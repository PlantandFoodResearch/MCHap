import numpy as np
from numba import jit

@jit(nopython=True)
def sum_log_prob(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    else:
        return y + np.log1p(np.exp(x - y))

@jit(nopython=True)
def rand_choice(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@jit(nopython=True)
def array_equal(x, y):
    """Assumes equal length and dtype.
    """
    n = len(x)
    for i in range(n):
        if x[i] != y[i]:
            return False
    return True


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
                        