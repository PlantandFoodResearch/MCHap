import numpy as np
import math
from numba import jit, njit

_FACTORIAL_LOOK_UP = np.fromiter((math.factorial(i) for i in range(21)), dtype=np.int64)

@njit
def factorial(x):
    if x in range(0, 21):
        return _FACTORIAL_LOOK_UP[x]
    else:
        raise ValueError('factorial functuion is only supported for values 0 to 20')


@njit
def interval_as_range(interval, max_range):
    if interval is None:
        return range(max_range)
    else:
        if len(interval) == 2:
            return range(interval[0], interval[1])
        else:
            raise ValueError('Interval must be `None` or array of length 2')

@njit
def random_interval(n, size_probs=None):
    if size_probs is None:
        # uniform distribution of positive sizes smaller than n
        size = np.random.randint(1, np.floor(n*1.3))
    else:
        # use provided distribution
        size = random_choice(size_probs)

    # center of the interval
    center = np.random.randint(0, n+1)
    
    lower = center - size // 2
    upper = center + size // 2

    if size % 2:
        # odd size so randomize longer end
        if np.random.random() < 0.5:
            lower -= 1
        else:
            upper += 1
    return max(lower, 0), min(upper, n)
    

@jit(nopython=True)
def _interval_inverse_mask(interval, n):
    if interval is None:
        mask = np.zeros(n, np.bool8)
    else:
        mask = np.ones(n, np.bool8)
        mask[interval[0]:interval[1]] = 0
    return mask

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


@njit
def log_likelihoods_as_conditionals(llks):
    # calculated denominator in log space
    n = len(llks)
    log_denominator = llks[0]
    for opt in range(1, n):
        log_denominator = sum_log_prob(log_denominator, llks[opt])

    # calculate conditional probabilities
    conditionals = np.empty(n)
    for opt in range(n):
        conditionals[opt] = np.exp(llks[opt] - log_denominator)
    return conditionals


@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
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


@jit(nopython=True)
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
    
    ploidy, n_base = genotype.shape
        
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


@jit(nopython=True)
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


@njit
def label_haplotypes(labels, genotype, interval=None):

    ploidy, n_base = genotype.shape
    labels[:] = 0

    r = interval_as_range(interval, n_base)

    for i in r:
        for j in range(1, ploidy):
            if genotype[j][i] == genotype[labels[j]][i]:
                # matches current assigned class
                pass
            else:
                # store previous assignment
                prev_label = labels[j]
                # assign to new label based on index
                # 'j' is the index and the label id
                labels[j] = j
                # check if following arrays match the new label
                for k in range(j + 1, ploidy):
                    if labels[k] == prev_label and genotype[j][i] == genotype[k][i]:
                        # this array is identical to the jth array
                        labels[k] = j


@njit
def haplotype_segment_labels(genotype, interval=None):
    """Create a labels matrix in whihe the first coloumn contains
    labels for haplotype segments within the specified range and
    the second column contains labels for the remander of the 
    haplotypes.

    If no interval is speciefied then the first column will contain
    labels for the full haplotypes and the second column will 
    contain zeros
    """
    ploidy, n_base = genotype.shape

    labels = np.zeros((ploidy, 2), np.int8)
    label_haplotypes(labels[:, 0], genotype, interval=interval)
    mask = _interval_inverse_mask(interval, n_base)
    label_haplotypes(labels[:, 1], genotype[:, mask], interval=None)
    return labels


@jit(nopython=True)
def conditional_probabilities(lnprobs):
    n = len(lnprobs)

    # calculated denominator in log space
    log_denominator = lnprobs[0]
    for i in range(1, n):
        log_denominator = sum_log_prob(log_denominator, lnprobs[i])

    # calculate conditional probabilities
    conditionals = np.empty(n)
    for i in range(n):
        conditionals[i] = np.exp(lnprobs[i] - log_denominator)
    
    return conditionals


@jit(nopython=True)
def structural_change(genotype, haplotype_indices, interval=None):
    """Mutate genotype by re-arranging haplotypes within a given interval.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    haplotype_indices : array_like, int, shape (ploidy)
        Indicies of haplotypes to use within the changed interval.
    interval : tuple, int
        Interval of base-positions to swap (defaults to all base positions).
    
    Returns
    -------
    None

    Notes
    -----
    Variables `genotype` is updated in place.

    """
    
    ploidy, n_base = genotype.shape
        
    cache = np.empty(ploidy, dtype=np.int8)

    r = interval_as_range(interval, n_base)
    
    for j in r:
        
        # copy to cache 
        for h in range(ploidy):
            cache[h] = genotype[h, j]
        
        # copy new bases back to genotype
        for h in range(ploidy):
            genotype[h, j] = cache[haplotype_indices[h]]


@jit(nopython=True)
def log_likelihood_structural_change(reads, genotype, haplotype_indices, interval=None):
    """Log likelihood of observed reads given a genotype given a structural change.

    Parameters
    ----------
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    haplotype_indices : array_like, int, shape (ploidy)
        Indicies of haplotypes to use within the changed interval.
    interval : tuple, int
        Interval of base-positions to swap (defaults to all base positions).

    Returns
    -------
    llk : float
        Log-likelihood of the observed reads given the genotype.
    """
    ploidy, n_base = genotype.shape
    n_reads = len(reads)
        
    intvl = interval_as_range(interval, n_base)
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                
                # check if in the altered region
                if j in intvl:
                    # use base from alternate hap
                    h_ = haplotype_indices[h]
                else:
                    # use base from current hap
                    h_ = h
                
                # get nucleotide index
                i = genotype[h_, j]

                read_hap_prod *= reads[r, j, i]
                
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk

