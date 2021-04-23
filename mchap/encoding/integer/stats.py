import numpy as np
from numba import njit

from mchap import mset
from mchap.encoding.integer.kmer import kmer_counts
from mchap.encoding.integer.sequence import depth
from mchap.assemble.util import array_equal


__all__ = [
    "minimum_error_correction",
    "read_assignment",
    "kmer_representation",
    "min_kmer_coverage",
]


def minimum_error_correction(read_calls, genotype):
    """
    Calculates the minimum error correction between
    reads and a genotype encoded as integers.

    Parameters
    ----------
    read_calls : ndarray, int, shape (n_reads, n_base)
        Array of integers encoding alleles.
    genotype : ndarray, int, shape (ploidy, n_base)
        Array of integers encoding alleles.

    Returns
    -------
    mec : ndarray, int, shape (n_reads,)
        Minimum error correction for each read.
    """
    read_calls = np.expand_dims(read_calls, 1)
    genotype = np.expand_dims(genotype, 0)
    diff = read_calls != genotype
    diff &= read_calls >= 0
    return diff.sum(axis=-1).min(axis=-1)


def read_assignment(read_calls, haplotypes):
    """
    Estimate the number of alleles assigned to each haplotype
    based on minimum error correction of individual reads.

    Parameters
    ----------
    read_calls : ndarray, int, shape (n_reads, n_base)
        Array of integers encoding alleles.
    haplotypes : ndarray, int, shape (ploidy, n_base)
        Array of integers encoding alleles.

    Returns
    -------
    assignment : ndarray, float, shape (n_reads, ploidy)
        Estimated haplotype assignment of each read.

    Notes
    -----
    If a reads minimum error correction score could result
    in an assignment to more than one haplotype then it is
    the assignment score is calculated as 1/n where n in
    the number of possible assignments.

    """
    read_calls = np.expand_dims(read_calls, 1)
    genotype = np.expand_dims(haplotypes, 0)
    diff = read_calls != genotype
    diff &= read_calls >= 0
    diff = diff.sum(axis=-1)
    mec = diff.min(axis=-1, keepdims=True)
    match = diff == mec
    return match / match.sum(axis=-1, keepdims=True)


def kmer_representation(read_calls, genotype, k=3):
    """Position-wise proportion of read-calls kmers which
    are present in haplotypes.

    Parameters
    ----------
    read_calls : ndarray, int, shape (n_reads, n_base)
        Array of integers encoding alleles.
    genotype : ndarray, int, shape (ploidy, n_base)
        Array of integers encoding alleles.
    k : int
        Size of kmers (default = 3).

    Result
    ------
    frequencies : ndarray, float, shape (n_base, )
        Proportion of read kmers present in haplotypes.

    """
    # create kmers and counts
    read_kmers, read_kmer_counts = kmer_counts(read_calls, k=k)
    hap_kmers, _ = kmer_counts(genotype, k=k)

    # handle case of no read kmers (i.e. from nan reads)
    if np.prod(read_kmers.shape) == 0:
        _, n_pos = hap_kmers.shape
        return np.ones(n_pos)

    # index of kmers not found in haplotypes
    idx = mset.count(hap_kmers, read_kmers) == 0

    # depth of unique kmers
    unique_depth = depth(read_kmers[idx], read_kmer_counts[idx])

    # depth of total kmers
    total_depth = depth(read_kmers, read_kmer_counts)

    # avoid divide by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        result = 1 - np.where(total_depth > 0, unique_depth / total_depth, 0)

    return result


@njit(cache=True)
def kmer_coverage(read_calls, genotype, k=3):
    ploidy, n_base = genotype.shape
    n_reads = len(read_calls)
    n_windows = n_base - (k - 1)
    covered = np.zeros(n_windows)
    total = np.zeros(n_windows)
    for w in range(n_windows):
        for r in range(n_reads):
            read_kmer = read_calls[r, w : w + k]
            if np.any(read_kmer < 0):
                pass
            else:
                total[w] += 1
                for h in range(ploidy):
                    hap_kmer = genotype[h, w : w + k]
                    if array_equal(hap_kmer, read_kmer):
                        covered[w] += 1
                        break

    return covered, total


def min_kmer_coverage(read_calls, genotype, ks):
    """Calculate the minimum number of read-kmers present in the genotype
    among all positions for a variety of levels of k.

    If the interval length is less than k or there are no complete kmers
    of length k then a nan value will be returned for that k.

    Parameters
    ----------
    read_calls : ndarray, int, shape (n_reads, n_base)
        Array of integers encoding alleles.
    genotype : ndarray, int, shape (ploidy, n_base)
        Array of integers encoding alleles.
    ks : ndarray : int, shape (n_ks, )
        Values of k.

    Result
    ------
    min_kmer_coverage : float, shape (n_ks, )
        Minimum coverage for each value of k.
    """
    n = len(ks)
    _, n_base = read_calls.shape
    out = np.zeros(n)
    for i in range(n):
        k = ks[i]
        if n_base < k:
            out[i] = np.nan
        else:
            num, denom = kmer_coverage(read_calls, genotype, k=k)
            if np.all(denom == 0):
                # no windows with kmers of this length
                out[i] = np.nan
            else:
                # ignore windows with no kmers
                with np.errstate(divide="ignore", invalid="ignore"):
                    out[i] = np.min(np.where(denom > 0, num / denom, 1))
    return out
