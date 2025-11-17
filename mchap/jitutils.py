import numpy as np
import math
import numba


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
    if x == y == -np.inf:
        return -np.inf
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
def ln_equivalent_permutations(dosage):
    """Natural long of the total number of equivalent genotype permutations
    based on the genotypes dosage.

    Parameters
    ----------
    dosage : ndarray, int
        1D vector of counts of each unique haplotype in a genotype.

    Notes
    -----
    A genotype is an unsorted multi-set of haplotypes hence rearranging the
    order of haplotypes in a (heterozygous) genotype can result in equivalent
    permutations

    """
    ploidy = np.sum(dosage)
    ln_num = math.lgamma(ploidy + 1)
    ln_denom = 0.0
    for i in range(len(dosage)):
        ln_denom += math.lgamma(dosage[i] + 1)
    return ln_num - ln_denom


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
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if k < 0:
        raise ValueError("k must be a non-negative integer")
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


_COMB_CACHE = np.zeros((100, 12), np.int64)
for n in range(_COMB_CACHE.shape[0]):
    for k in range(_COMB_CACHE.shape[1]):
        _COMB_CACHE[n, k] = _comb(n, k)


@numba.njit(cache=True)
def comb(n: int, k: int) -> int:
    cache_n, cache_k = _COMB_CACHE.shape
    if (n < cache_n) and (k < cache_k):
        return _COMB_CACHE[n, k]
    else:
        return _comb(n, k)


@numba.njit(cache=True)
def _comb_with_replacement(n: int, k: int) -> int:
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0 and k == 0:
        return 0
    n = n + k - 1
    return comb(n, k)


_COMB_WITH_REPLACEMENT_CACHE = np.zeros((100, 12), np.int64)
for n in range(_COMB_WITH_REPLACEMENT_CACHE.shape[0]):
    for k in range(_COMB_WITH_REPLACEMENT_CACHE.shape[1]):
        _COMB_WITH_REPLACEMENT_CACHE[n, k] = _comb_with_replacement(n, k)


@numba.njit(cache=True)
def comb_with_replacement(n: int, k: int) -> int:
    cache_n, cache_k = _COMB_WITH_REPLACEMENT_CACHE.shape
    if (n < cache_n) and (k < cache_k):
        return _COMB_WITH_REPLACEMENT_CACHE[n, k]
    else:
        return _comb_with_replacement(n, k)


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
            index += comb_with_replacement(a, i + 1)
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
            new = comb_with_replacement(n, p)
        n -= 1
        remainder -= prev
        out[p - 1] = n
    return out


@numba.njit(cache=True)
def array_equal(x, y, interval=None):
    """Check if two one-dimentional integer arrays are equal.

    Parameters
    ----------
    x, y : ndarray, int
        1D vectors of integers.
    interval : tuple, int shape (2, ), optional
        A pair of integers defining a half open interval.

    Returns
    -------
    equality : bool
        True if `x` and `y` are equal (within an interval
        if specified).

    """
    if interval is None:
        r = range(len(x))
    else:
        r = range(interval[0], interval[1])

    for i in r:
        if x[i] != y[i]:
            return False
    return True


@numba.njit(cache=True)
def count_haplotype_copies(genotype, h):
    """Count the number of copies of the specified haplotype
    within a genotype.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        A genotype of haplotype alleles encoded as integers.
    h : int
        Index of the haplotype to count.

    Returns
    -------
    count : int
        The number of haplotype copies.
    """
    ploidy = len(genotype)
    count = 1
    for i in range(ploidy):
        if i == h:
            pass
        else:
            if array_equal(genotype[i], genotype[h]):
                count += 1
    return count


@numba.njit(cache=True)
def get_haplotype_dosage(dosage, genotype, interval=None):
    """Calculates the dosage of a set of integer encoded haplotypes by
    checking for array equality.

    Parameters
    ----------
    dosage : ndarray, int, shape (ploidy)
        Array to update with dosage of each haplotype.
    genotype : ndarray, int, shape (ploidy, n_base)
        Initial state of haplotypes with base positions encoded as
        simple integers.

    Returns
    -------
    None

    Notes
    -----
    The `dosage` variable is updated in place.
    The `dosage` array should always sum to the number of haplotypes
    in the `genotype`.
    A value of `0` in the `dosage` array indicates that that haplotype
    is a duplicate of another.

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
            for p in range(h + 1, ploidy):
                if dosage[p] == 0:
                    # this haplotype has already been identified as equal to another
                    pass
                else:
                    if array_equal(genotype[h], genotype[p], interval=interval):
                        dosage[h] += 1
                        dosage[p] = 0


@numba.njit(cache=True)
def set_haplotype_dosage(genotype, dosage):
    """Set a genotype of haplotype arrays to a new dosage.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Initial state of haplotypes with alleles at each base
        positions encoded as integers.
    dosage : ndarray, int, shape (ploidy)
        Array with dose of each haplotype.

    Returns
    -------
    None

    Notes
    -----
    The `dosage` variable is updated in place.
    The `dosage` array should always sum to the number of
    haplotypes in the `genotype`.

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


@numba.njit(cache=True)
def sample_snv_alleles(array, dtype=np.int8):
    """Sample a random set of SNV alleles from probabilities.

    Parameters
    ----------
    array : ndarray, float, shape (n_read, n_base, max_allele)
        Probability of each allele at each base position
    dtype : type
        Numpy dtype of alleles.

    Returns
    -------
    alleles : ndarray, int, shape (n_read, n_base)
        Sampled alleles

    Notes
    -----
    Does not handle gaps (nan values).

    """
    # normalize
    shape = array.shape[0:-1]

    array = array.reshape(-1, array.shape[-1])
    sums = np.sum(array, axis=-1)
    dists = array / np.expand_dims(sums, -1)
    n = len(dists)

    alleles = np.empty(n, dtype=dtype)

    for i in range(n):
        alleles[i] = random_choice(dists[i])

    return alleles.reshape(shape)


@numba.njit(cache=True)
def structural_change(genotype, haplotype_indices, interval=None):
    """Mutate genotype by re-arranging haplotypes
    within a given interval.

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_base)
        Set of haplotypes with base positions encoded as
        simple integers from 0 to n_allele.
    haplotype_indices : ndarray, int, shape (ploidy)
        Indicies of haplotypes to update alleles from.
    interval : tuple, int, optional
        If set then base-positions copies/swaps between
        haplotype is constrained to the specified
        half open interval (defaults = None).

    Returns
    -------
    None

    Notes
    -----
    Variable `genotype` is updated in place.

    """

    ploidy, n_base = genotype.shape

    cache = np.empty(ploidy, dtype=np.int8)

    if interval is None:
        r = range(n_base)
    else:
        r = range(interval[0], interval[1])

    for j in r:
        # copy to cache
        for h in range(ploidy):
            cache[h] = genotype[h, j]

        # copy new bases back to genotype
        for h in range(ploidy):
            genotype[h, j] = cache[haplotype_indices[h]]
