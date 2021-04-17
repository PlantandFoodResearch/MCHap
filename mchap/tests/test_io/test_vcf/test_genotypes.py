import numpy as np

from mchap.io import vcf
from mchap.testing import simulate_reads
from mchap.assemble.likelihood import log_likelihood


def test_genotype_string():
    haplotypes = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    genotype = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )

    expect = "0/2/2/."
    actual = vcf.genotype_string(genotype, haplotypes)
    assert actual == expect


def test_sort_haplotypes__ref():
    genotypes = np.array(
        [
            [[0, 1, 1], [0, 1, 1], [1, 1, 1], [-1, -1, -1]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
        ],
        np.int8,
    )
    actual, counts = vcf.sort_haplotypes(genotypes)

    expect = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 1],
        ],
        np.int8,
    )
    expect_counts = np.array([4, 5, 2])

    np.testing.assert_array_equal(actual, expect)
    np.testing.assert_array_equal(counts, expect_counts)


def test_sort_haplotypes__no_ref():
    genotypes = np.array(
        [
            [[0, 1, 1], [0, 1, 1], [1, 1, 1], [-1, -1, -1]],
            [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
            [[1, 0, 0], [1, 0, 0], [1, 1, 1], [1, 1, 1]],
        ],
        np.int8,
    )
    actual, counts = vcf.sort_haplotypes(genotypes)

    expect = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
            [0, 1, 1],
        ],
        np.int8,
    )
    expect_counts = np.array([0, 5, 4, 2])

    np.testing.assert_array_equal(actual, expect)
    np.testing.assert_array_equal(counts, expect_counts)


def test_expected_dosage():
    haplotypes = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    genotypes = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 1, 1], [0, 1, 1]],
        ],
        dtype=np.int8,
    )
    probs = np.array([0.9, 0.1])

    actual = vcf.expected_dosage(genotypes, probs, haplotypes)
    expect = np.array([2.9, 1.1])
    np.testing.assert_almost_equal(actual, expect)


def test_genotype_likelihoods():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 1, 0, 1, 1, 1],  # 1
            [1, 1, 1, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
        ]
    )
    genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 1, 1, 2],  # true genotype
            [1, 1, 1, 2],
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 2, 2, 2],
            [1, 2, 2, 2],
            [2, 2, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 3],
            [0, 1, 1, 3],
            [1, 1, 1, 3],
            [0, 0, 2, 3],
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [0, 2, 2, 3],
            [1, 2, 2, 3],
            [2, 2, 2, 3],
            [0, 0, 3, 3],
            [0, 1, 3, 3],
            [1, 1, 3, 3],
            [0, 2, 3, 3],
            [1, 2, 3, 3],
            [2, 2, 3, 3],
            [0, 3, 3, 3],
            [1, 3, 3, 3],
            [2, 3, 3, 3],
            [3, 3, 3, 3],
        ]
    )
    correct_genotype = genotypes[7]
    ploidy = len(correct_genotype)
    genotype_haps = haplotypes[correct_genotype]
    error_haps = haplotypes[[3]]
    reads_correct = simulate_reads(
        genotype_haps,
        n_reads=16,
        uniform_sample=True,
        errors=False,
        qual=(60, 60),
    )
    reads_error = simulate_reads(
        error_haps,
        n_reads=1,
        uniform_sample=True,
        errors=False,
        qual=(60, 60),
    )
    _, n_pos, n_nucl = reads_correct.shape
    reads = np.concatenate([reads_correct, reads_error]).reshape(-1, n_pos, n_nucl)
    expect = np.empty(len(genotypes))
    for i in range(len(expect)):
        expect[i] = log_likelihood(reads, haplotypes[genotypes[i]])
    expect = np.log10(np.exp(expect))
    actual = vcf.genotype_likelihoods(reads, ploidy, haplotypes)
    np.testing.assert_almost_equal(expect, actual, decimal=5)
