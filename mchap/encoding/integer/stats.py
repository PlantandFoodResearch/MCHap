import numpy as np

from mchap import mset
from mchap.encoding.integer.kmer import kmer_counts
from mchap.encoding.integer.sequence import depth


__all__ = [
    "minimum_error_correction",
    "read_assignment",
    "kmer_representation",
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
