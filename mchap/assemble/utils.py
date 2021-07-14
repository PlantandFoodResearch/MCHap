import numpy as np
import numba

from mchap.jitutils import random_choice


@numba.njit(cache=True)
def interval_as_range(interval, max_range):
    # TODO: inline this into the callers and remove
    if interval is None:
        return range(max_range)
    else:
        if len(interval) == 2:
            return range(interval[0], interval[1])
        else:
            raise ValueError("Interval must be `None` or array of length 2")


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
    r = interval_as_range(interval, len(x))

    for i in r:
        if x[i] != y[i]:
            return False
    return True


@numba.njit(cache=True)
def count_haplotype_copies(genotype, h):
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
def get_dosage(dosage, genotype, interval=None):
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
def set_dosage(genotype, dosage):
    """Set a genotype to a new dosage.

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
def sample_alleles(array, dtype=np.int8):
    """Sample a random set of alleles from probabilities.

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

    r = interval_as_range(interval, n_base)

    for j in r:

        # copy to cache
        for h in range(ploidy):
            cache[h] = genotype[h, j]

        # copy new bases back to genotype
        for h in range(ploidy):
            genotype[h, j] = cache[haplotype_indices[h]]
