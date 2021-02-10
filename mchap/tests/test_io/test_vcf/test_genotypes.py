import numpy as np

from mchap.io import vcf


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
