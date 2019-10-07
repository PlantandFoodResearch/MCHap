import numpy as np
from numba import jit


@jit(nopython=True)
def sum_log_prob(x, y):
    """Sum of probabilities in log space.

    Parameters
    ----------
    x : float
        A log-transformed probability.
    y : float
        A log-transformed probability.

    Returns
    -------
    z : The log-transformed sum of the un-transformed `x` and `y`.

    """
    if x > y:
        return x + np.log1p(np.exp(y - x))
    else:
        return y + np.log1p(np.exp(x - y))


@jit(nopython=True)
def random_choice(probabilities):
    """Random choice of options given a set of probabilities.

    Parameters
    ----------
    options : array_like, int
        An array of options to choose from.
    probabilities : array_like, float
        An array of probabilities asociated with each value in `options`.

    Returns
    -------
    choice : int
        A value from `options` selected at random with using `probabilities`.

    """
    return np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right")


@jit(nopython=True)
def greedy_choice(options, probabilities):
    """Greedy choice of options given a set of probabilities.

    Parameters
    ----------
    options : array_like, int
        An array of options to choose from.
    probabilities : array_like, float
        An array of probabilities asociated with each value in `options`.

    Returns
    -------
    choice : int
        The value in `options` with the largest corresponding value in `probabilities`.

    """
    return options[np.argmax(probabilities)]


@jit(nopython=True)
def array_equal(x, y):
    """Check if two one-dimentional integer arrays are equal.

    Parameters
    ----------
    x : array_like, int
        A one-dimentional array of integers.
    y : array_like, int
        A one-dimentional array of integers with the same length as `x`.

    Returns
    -------
    equality : bool
        True if `x` and `y` are equal, else False.

    """
    n = len(x)
    for i in range(n):
        if x[i] != y[i]:
            return False
    return True


@jit(nopython=True)
def set_dosage_to_genotype(genotype, dosage):
    """Calculates the dosage of a set of integer encoded haplotypes by 
    checking for array equality.
    
    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    dosage : array_like, int, shape (ploidy)
        Array to update with dosage of each haplotype.

    Returns
    -------
    None
    
    Notes
    -----
    The `dosage` variable is updated in place.
    The `dosage` array should always sum to the number of haplotypes in the `genotype`.
    A value of `0` in the `dosage` array indicates that that haplotype is a duplicate of another.

    """
    # start with assumption that all are unique
    dosage[:] = 1
    
    ploidy, _ = genotype.shape
    
    for h in range(ploidy):
        if dosage[h] == 0:
            # this haplotype has already been identified as equal to another
            pass
        else:
            # iterate through remaining haps
            for p in range(h+1, ploidy):
                if dosage[p] == 0:
                    # this haplotype has already been identified as equal to another
                    pass
                else:
                    if array_equal(genotype[h], genotype[p]):
                        dosage[h] += 1
                        dosage[p] = 0


@jit(nopython=True)
def set_genotype_to_dosage(genotype, dosage):
    """Set a genotype to a new dosage.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    dosage : array_like, int, shape (ploidy)
        Array with dose of each haplotype.

    Returns
    -------
    None
    
    Notes
    -----
    The `dosage` variable is updated in place.
    The `dosage` array should always sum to the number of haplotypes in the `genotype`.

    """    
    dosage = dosage.copy()

    ploidy = len(genotype)
    
    h_y = 0
    
    for h_x in range(ploidy):

        while dosage[h_x] > 1:
            
            # don't iter over dosages we know are no longer 0
            for h_y in range(h_y, ploidy):
                
                if dosage[h_y] == 0:
                    
                    genotype[h_y] = genotype[h_x]
                    dosage[h_x] -= 1
                    dosage[h_y] += 1
