import numpy as np


def minimum_error_correction(reads, genotype):
    """
    Calculates the minimum error correction between
    reads and a genotype encoded as integers.

    Parameters
    ----------
    reads : ndarray, int, shape (n_reads, n_base)
        Array of integers encoding alleles.
    genotype : ndarray, int, shape (ploidy, n_base)
        Array of integers encoding alleles.

    Returns
    -------
    mec : ndarray, int, shape (n_reads,)
        Minimum error correction for each read.
    """
    reads = np.expand_dims(reads, 1)
    genotype = np.expand_dims(genotype, 0)
    diff = reads != genotype
    diff &= reads >= 0
    return diff.sum(axis=-1).min(axis=-1)


def read_assignment(reads, genotype):
    """
    Estimate the number of alleles assigned to each haplotype
    based on minimum error correction of individual reads.

    Parameters
    ----------
    reads : ndarray, int, shape (n_reads, n_base)
        Array of integers encoding alleles.
    genotype : ndarray, int, shape (ploidy, n_base)
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
    reads = np.expand_dims(reads, 1)
    genotype = np.expand_dims(genotype, 0)
    diff = reads != genotype
    diff &= reads >= 0
    diff = diff.sum(axis=-1)
    mec = diff.min(axis=-1, keepdims=True)
    match = diff == mec
    return match / match.sum(axis=-1, keepdims=True)

