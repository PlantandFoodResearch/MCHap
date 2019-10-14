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
def array_equal(x, y, interval=(0, -1)):
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
    if interval == (0, -1):
        interval = (0, len(x))
    
    for i in range(interval[0], interval[1]):
        if x[i] != y[i]:
            return False
    return True


@jit(nopython=True)
def get_dosage(dosage, genotype, interval=(0, -1)):
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
    
    if interval == (0, -1):
        interval = (0, n_base)
    
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
def overwrite(genotype, h_x, h_y, interval=(0, -1)):
    """Overwrite (sub-)haplotype y with values from (sub-)haplotype x of a genotype.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    h_x : int
        Index of haplotype x in genotype to recombine with haplotype y.
    h_y : int
        Index of haplotype y in genotype to recombine with haplotype x.
    interval : tuple, int
        Interval of base-positions to overwrite (defaults to all base positions).
    
    Returns
    -------
    None

    Notes
    -----
    Variables `genotype` is updated in place.

    """
    
    if interval == (0, -1):
        interval = (0, genotype.shape[-1])
    
    # swap bases from the recombination point
    for j in range(interval[0], interval[1]):
        
        genotype[h_y, j] = genotype[h_x, j]


@jit(nopython=True)
def swap(genotype, h_x, h_y, interval=(0, -1)):
    """Swap a pair of (sub-)haplotypes within a genotype.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    h_x : int
        Index of haplotype x in genotype to recombine with haplotype y.
    h_y : int
        Index of haplotype y in genotype to recombine with haplotype x.
    interval : tuple, int
        Interval of base-positions to swap (defaults to all base positions).
    
    Returns
    -------
    None

    Notes
    -----
    Variables `genotype` is updated in place.

    """
    
    if interval == (0, -1):
        interval = (0, genotype.shape[-1])
    
    # swap bases from the recombination point
    for j in range(interval[0], interval[1]):
        
        j_x = genotype[h_x, j]
        j_y = genotype[h_y, j]
        
        genotype[h_x, j] = j_y
        genotype[h_y, j] = j_x


@jit(nopython=True)
def structural_change(genotype, haplotypes, interval=(0, -1)):
    """Mutate genotype by re-arranging haplotypes within a given interval.

    Parameters
    ----------
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    haplotypes : array_like, int, shape (ploidy)
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
    
    if interval == (0, -1):
        interval = (0, n_base)
    
    cache = np.empty(ploidy, dtype=np.int8)
    
    for j in range(interval[0], interval[1]):
        
        # copy to cache 
        for h in range(ploidy):
            cache[h] = genotype[h, j]
        
        # copy new bases back to genotype
        for h in range(ploidy):
            genotype[h, j] = cache[haplotypes[h]]


@jit(nopython=True)
def log_likelihood_structural_change(reads, genotype, haplotypes, interval=(0, -1)):
    """Log likelihood of observed reads given a genotype given a structural change.

    Parameters
    ----------
    reads : array_like, float, shape (n_reads, n_base, n_nucl)
        Observed reads encoded as an array of probabilistic matrices.
    genotype : array_like, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as simple integers from 0 to n_nucl.
    haplotypes : array_like, int, shape (ploidy)
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
    
    if interval == (0, -1):
        interval = (0, n_base)
    
    interval_ = range(interval[0], interval[1])
       
    llk = 0.0
    
    for r in range(n_reads):
        
        read_prob = 0
        
        for h in range(ploidy):
            read_hap_prod = 1.0
            
            for j in range(n_base):
                
                # check if in the altered region
                if j in interval_:
                    # use base from alternate hap
                    h_ = haplotypes[h]
                else:
                    # use base from current hap
                    h_ = h
                
                # get nucleotide index
                i = genotype[h_, j]

                read_hap_prod *= reads[r, j, i]
                
            read_prob += read_hap_prod/ploidy
        
        llk += np.log(read_prob)
                    
    return llk

