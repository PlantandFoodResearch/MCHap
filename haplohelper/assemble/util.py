import numpy as np
import math
import numba

_FACTORIAL_LOOK_UP = np.fromiter((math.factorial(i) for i in range(21)), dtype=np.int64)


@numba.njit
def interval_as_range(interval, max_range):
    if interval is None:
        return range(max_range)
    else:
        if len(interval) == 2:
            return range(interval[0], interval[1])
        else:
            raise ValueError('Interval must be `None` or array of length 2')


@numba.njit
def add_log_prob(x, y):
    """Sum of two probabilities in log space.

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


@numba.njit
def sum_log_probs(array):
    acumulate = array[0]
    for i in range(1, len(array)):
        acumulate = add_log_prob(acumulate, array[i])
    return acumulate


@numba.njit
def log_likelihoods_as_conditionals(llks):
    # calculated denominator in log space
    log_denominator = sum_log_probs(llks)

    # calculate conditional probabilities
    n = len(llks)
    conditionals = np.empty(n)
    for opt in range(n):
        conditionals[opt] = np.exp(llks[opt] - log_denominator)
    return conditionals


@numba.njit
def random_choice(probabilities):
    """Random choice of options given a set of probabilities.

    Parameters
    ----------
    probabilities : array_like, float
        An array of probabilities that sum to one.

    Returns
    -------
    choice : int
        The index of a randomly selected probability.

    """
    return np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right")


@numba.njit
def greedy_choice(probabilities):
    """Greedy choice of options given a set of probabilities.

    Parameters
    ----------
    probabilities : array_like, float
        An array of probabilities that sum to one.

    Returns
    -------
    choice : int
        The index of the largest probability.

    """
    return np.argmax(probabilities)


@numba.njit
def array_equal(x, y, interval=None):
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
    r = interval_as_range(interval, len(x))
    
    for i in r:
        if x[i] != y[i]:
            return False
    return True


@numba.njit
def get_dosage(dosage, genotype, interval=None):
    """Calculates the dosage of a set of integer encoded haplotypes by 
    checking for array equality.
    
    Parameters
    ----------
    dosage : array_like, int, shape (ploidy)
        Array to update with dosage of each haplotype.
    genotype : array_like, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as simple integers from 0 to n_nucl.

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
                    if array_equal(genotype[h], genotype[p], interval=interval):
                        dosage[h] += 1
                        dosage[p] = 0


@numba.njit
def set_dosage(genotype, dosage):
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


@numba.njit
def factorial_20(x):
    if x in range(0, 21):
        return _FACTORIAL_LOOK_UP[x]
    else:
        raise ValueError('factorial functuion is only supported for values 0 to 20')


@numba.njit
def count_equivalent_permutations(dosage):
    """Counts the total number of equivilent genotype perterbation based on the dosage.

    A genotype is an unsorted set of haplotypes hence the genotype `{A, B}` is equivielnt
    to the genotype `{B, A}`.
    A fully homozygous genotype e.g. `{A, A}` has only one possible perterbation.
    """
    ploidy = np.sum(dosage)
    numerator = factorial_20(ploidy)
    denominator = 1
    for i in range(len(dosage)):
        denominator *= factorial_20(dosage[i])
    return numerator // denominator
