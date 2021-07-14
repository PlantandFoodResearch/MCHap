import numpy as np
import math
import numba

_FACTORIAL_LOOK_UP = np.fromiter((math.factorial(i) for i in range(21)), dtype=np.int64)


@numba.njit(cache=True)
def factorial_20(x):
    """Returns the factorial of integers in the range [0, 20] (inclusive)

    Parameters
    ----------
    x : int
        An integer

    Returns
    -------
    x_fac : int
        Factorial of x

    """
    if x in range(0, 21):
        return _FACTORIAL_LOOK_UP[x]
    else:
        raise ValueError("factorial functuion is only supported for values 0 to 20")


@numba.njit(cache=True)
def add_log_prob(x, y):
    """Sum of two probabilities in log space.

    Parameters
    ----------
    x, y : float
        Log-transformed probabilities.

    Returns
    -------
    z : float
        The log-transformed sum of the un-transformed `x` and `y`.

    """
    if x > y:
        return x + np.log1p(np.exp(y - x))
    else:
        return y + np.log1p(np.exp(x - y))


@numba.njit(cache=True)
def sum_log_probs(array):
    """Sum of values in log space.

    Parameters
    ----------
    array : ndarray, float, shape (n_values, )
        Log-transformed values.

    Returns
    -------
    z : float
        The log-transformed sum of the un-transformed values.

    """
    acumulate = array[0]
    for i in range(1, len(array)):
        acumulate = add_log_prob(acumulate, array[i])
    return acumulate


@numba.njit(cache=True)
def normalise_log_probs(llks):
    """Returns normalised probabilities of
    an array of log-transformed probabilities.

    Parameters
    ----------
    llks : ndarray, float, shape (n_values, )
        Log-transformed likelihoods.

    Returns
    -------
    conditionals : ndarray, float, shape (n_values, )
        Normalised conditional probabilities.

    """
    # calculated denominator in log space
    log_denominator = sum_log_probs(llks)

    # calculate conditional probabilities
    n = len(llks)
    normalised = np.empty(n)
    for opt in range(n):
        normalised[opt] = np.exp(llks[opt] - log_denominator)
    return normalised


@numba.njit(cache=True)
def random_choice(probabilities):
    """Random choice of options given a set of probabilities.

    Parameters
    ----------
    probabilities : ndarray, float, shape (n_values, )
        1D vector of probabilities that sum to one.

    Returns
    -------
    choice : int
        The index of a randomly selected probability.

    """
    return np.searchsorted(np.cumsum(probabilities), np.random.random(), side="right")


@numba.njit(cache=True)
def greedy_choice(probabilities):
    """Greedy choice of options given a set of probabilities.

    Parameters
    ----------
    probabilities : ndarray, float, shape (n_values, )
        1D vector of probabilities that sum to one.

    Returns
    -------
    choice : int
        The index of the largest probability.

    """
    return np.argmax(probabilities)


@numba.njit(cache=True)
def increment_genotype(genotype):
    """Increment a genotype of allele numbers to the next genotype
    in VCF sort order.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy,)
        Array of allele numbers in the genotype.

    Notes
    -----
    Mutates genotype array in place.
    """
    ploidy = len(genotype)
    if ploidy == 1:
        # haploid case
        genotype[0] += 1
        return
    previous = genotype[0]
    for i in range(1, ploidy):
        allele = genotype[i]
        if allele == previous:
            pass
        elif allele > previous:
            i -= 1
            genotype[i] += 1
            genotype[0:i] = 0
            return
        else:
            raise ValueError("genotype alleles are not in ascending order")
    # all alleles are equal
    genotype[-1] += 1
    genotype[0:-1] = 0


@numba.njit(cache=True)
def n_choose_k(n, k):
    """Calculate n choose k for values of n and k < 20.
    Parameters
    ----------
    n : int
        Number of elements to choose from.
    k : int
        Number of elements to be drawn.

    Returns
    -------
    combinations : int
        Number of possible combinations of size k drawn from
        a set of size n.

    Notes
    -----
    Formula: (n!) / (k!(n-k)!)

    """
    return factorial_20(n) // (factorial_20(k) * factorial_20(n - k))


@numba.njit(cache=True)
def count_equivalent_permutations(dosage):
    """Counts the total number of equivilent genotype perterbations
    based on the genotypes dosage.

    Parameters
    ----------
    dosage : ndarray, int
        1D vector of counts of each unique haplotype in a genotype.

    Notes
    -----
    A genotype is an unsorted multi-set of haplotypes hence rearanging the
    order of haplotypes in a (heterozygous) genotype can result in equivilent
    permutations

    """
    ploidy = np.sum(dosage)
    numerator = factorial_20(ploidy)
    denominator = 1
    for i in range(len(dosage)):
        denominator *= factorial_20(dosage[i])
    return numerator // denominator


@numba.njit(cache=True)
def natural_log_to_log10(x):
    e = np.exp(1)
    return x * np.log10(e)


@numba.njit(cache=True)
def seed_numba(seed):
    """Set numba random seed"""
    np.random.seed(seed)


@numba.njit(cache=True)
def _greatest_common_denominatior(x: int, y: int) -> int:
    while y != 0:
        t = x % y
        x = y
        y = t
    return x


@numba.njit(cache=True)
def _comb(n: int, k: int) -> int:
    if k > n:
        return 0
    r = 1
    for d in range(1, k + 1):
        gcd = _greatest_common_denominatior(r, d)
        r //= gcd
        r *= n
        r //= d // gcd
        n -= 1
    return r


@numba.njit(cache=True)
def _comb_with_replacement(n: int, k: int) -> int:
    n = n + k - 1
    return _comb(n, k)


@numba.njit(cache=True)
def genotype_alleles_as_index(alleles):
    """Convert genotypes to the index of their array position
    following the VCF specification for fields of length G.

    Parameters
    ----------
    alleles
        Integer alleles of the genotype.

    Returns
    -------
    index
        Index of genotype following the sort order described in the
        VCF spec.
    """
    index = 0
    for i in range(len(alleles)):
        a = alleles[i]
        if a >= 0:
            index += _comb_with_replacement(a, i + 1)
        elif a < 0:
            raise ValueError("Allele numbers must be >= 0.")
    return index


@numba.njit(cache=True)
def index_as_genotype_alleles(index, ploidy):
    """Convert the index of a genotype sort position to the
    genotype call indicated by that index following the VCF
    specification for fields of length G.

    Parameters
    ----------
    index
        Index of genotype following the sort order described in the
        VCF spec. An index less than 0 is invalid and will return an
        uncalled genotype.
    ploidy
        Ploidy of the genotype.

    Returns
    -------
    alleles
        Integer alleles of the genotype.
    """
    out = np.full(ploidy, -2, np.int64)
    if index < 0:
        # handle non-call
        out[:ploidy] = -1
        return
    remainder = index
    for index in range(ploidy):
        # find allele n for position k
        p = ploidy - index
        n = -1
        new = 0
        prev = 0
        while new <= remainder:
            n += 1
            prev = new
            new = _comb_with_replacement(n, p)
        n -= 1
        remainder -= prev
        out[p - 1] = n
    return out
