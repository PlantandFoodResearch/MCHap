import numpy as np
import pytest

from mchap.assemble.likelihood import *


def reference_likelihood(reads, genotype):
    """Reference implementation of likelihood function.
    """
    n_reads, n_pos, n_nucl = reads.shape
    ploidy, genotype_n_pos = genotype.shape

    # same number of base/variant positions in reads and haplotypes
    assert genotype_n_pos == n_pos

    # convert genotype to matrix of onehot row-vectors
    genotype_matrix = np.zeros((ploidy, n_pos, n_nucl), dtype=reads.dtype)

    for h in range(ploidy):
        for j in range(n_pos):
            i = genotype[h, j]
            # allele is the index of the 1 in the one-hot vector
            genotype_matrix[h, j, i] = 1

    # expand dimentions to use broadcasting
    reads = reads.reshape((n_reads, 1, n_pos, n_nucl))
    genotype = genotype_matrix.reshape((1, ploidy, n_pos, n_nucl))

    # probability of match for each posible allele at each position
    # for every read-haplotype combination
    probs = reads * genotype

    # probability of match at each position for every read-haplotype combination
    # any nans along this axis indicate that ther is a gap in the read at this position
    probs = np.sum(probs, axis=-1)

    # joint probability of match at all positions for every read-haplotype combination
    # nans caused by read gaps should be ignored
    probs = np.nanprod(probs, axis=-1)

    # probability of a read being generated from a genotype is the mean of
    # probabilities of that read matching each haplotype within the genotype
    # this assumes that the reads are generated in equal probortions from each genotype
    probs = np.mean(probs, axis=-1)

    # the probability of a set of reads being generated from a genotype is the 
    # joint probability of all reads within the set being generated from that genotype
    probs = np.prod(probs)

    # the likelihood is P(reads|genotype)
    return probs


def test_log_likelihood():
    reads = np.array([
        [[0.8, 0.2],
         [0.8, 0.2],
         [0.8, 0.2]],
        [[0.8, 0.2],
         [0.8, 0.2],
         [0.2, 0.8]],
        [[0.8, 0.2],
         [0.8, 0.2],
         [np.nan, np.nan]]
    ])

    genotype = np.array([
        [0, 0, 0],
        [0, 0, 1]
    ], dtype=np.int8)

    query = log_likelihood(reads, genotype)
    answer = np.log(reference_likelihood(reads, genotype))

    assert np.round(query, 10) == np.round(answer, 10)


@pytest.mark.parametrize("reads, genotype, haplotype_indices, interval, final_genotype", [
    pytest.param(
        [
            [[0.8, 0.2],
             [0.8, 0.2],
             [0.8, 0.2]],
            [[0.8, 0.2],
             [0.8, 0.2],
             [0.2, 0.8]],
            [[0.8, 0.2],
             [0.8, 0.2],
             [np.nan, np.nan]]
        ],
        [
            [0, 0, 0],
            [0, 0, 1]
        ],
        [0, 1],
        None,
        [
            [0, 0, 0],
            [0, 0, 1]
        ],
        id='2x-no-change'
    ),
    pytest.param(
        [
            [[0.8, 0.2],
             [0.8, 0.2],
             [0.8, 0.2]],
            [[0.8, 0.2],
             [0.8, 0.2],
             [0.2, 0.8]],
            [[0.8, 0.2],
             [0.8, 0.2],
             [np.nan, np.nan]]
        ],
        [
            [0, 0, 0],
            [0, 0, 1]
        ],
        [1, 1],
        None,
        [
            [0, 0, 1],
            [0, 0, 1]
        ],
        id='2x-overwrite'
    ),
])
def test_log_likelihood_structural_change(reads, genotype, haplotype_indices, interval, final_genotype):

    reads = np.array(reads, dtype=float)
    genotype = np.array(genotype, dtype=np.int8)
    haplotype_indices = np.array(haplotype_indices, dtype=int)
    final_genotype = np.array(final_genotype, dtype=np.int8)

    query = log_likelihood_structural_change(reads, genotype, haplotype_indices, interval=interval)
    answer = log_likelihood(reads, final_genotype)
    reference =  np.log(reference_likelihood(reads, final_genotype))

    assert query == answer
    assert np.round(query, 10) == np.round(reference, 10)
