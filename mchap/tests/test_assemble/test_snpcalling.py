import numpy as np
import pytest

from mchap.assemble.snpcalling import snp_posterior


def test_snp_posterior__zero_reads():
    reads = np.empty((0, 2, 2), dtype=float)

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :], n_alleles=2, ploidy=4, inbreeding=0.0
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs)


def test_snp_posterior__zero_reads__inbred():
    reads = np.empty((0, 2, 2), dtype=float)

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.1015625, 0.24375, 0.309375, 0.24375, 0.1015625])

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :], n_alleles=2, ploidy=4, inbreeding=0.1
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=5)


def test_snp_posterior__nan_reads():

    read = np.array(
        [
            [
                [np.nan, np.nan],
                [0.9, 0.1],
            ]
        ],
        dtype=float,
    )
    reads = np.tile(read, (3, 1, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :], n_alleles=2, ploidy=4, inbreeding=0.0
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs)


def test_snp_posterior__nan_reads__inbred():

    read = np.array(
        [
            [
                [np.nan, np.nan],
                [0.9, 0.1],
            ]
        ],
        dtype=float,
    )
    reads = np.tile(read, (3, 1, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.1015625, 0.24375, 0.309375, 0.24375, 0.1015625])

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :], n_alleles=2, ploidy=4, inbreeding=0.1
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=5)


def test_snp_posterior__novel_allele():
    # this tests for a rare edge case in which the allele in the sample
    # is no a known reference or alternate allele
    # this results in all known alleles having very small probabilities
    # which (with a high read depth) can cause issues if probabilities
    # are not handled in their log form
    read = np.array(
        [
            [
                [0.00001, 0.00001],  # target snp
                [0.99999, 0.00001],
            ]
        ],
        dtype=float,
    )
    reads = np.tile(read, (1000, 1, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([0.0625, 0.25, 0.375, 0.25, 0.0625])

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :], n_alleles=2, ploidy=4, inbreeding=0.0
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs)


@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
def test_snp_posterior__homozygous_deep(use_read_counts):

    read = np.array(
        [
            [
                [0.99999, 0.00001],  # target snp
                [0.99999, 0.00001],
            ]
        ],
        dtype=float,
    )
    if use_read_counts:
        read_counts = np.array([100])
        reads = np.tile(read, (1, 1, 1))
    else:
        read_counts = None
        reads = np.tile(read, (100, 1, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :],
        n_alleles=2,
        ploidy=4,
        inbreeding=0.0,
        read_counts=read_counts,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=10)


@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
def test_snp_posterior__homozygous_shallow(use_read_counts):

    read = np.array(
        [
            [
                [0.99999, 0.00001],  # target snp
                [0.99999, 0.00001],
            ]
        ],
        dtype=float,
    )
    if use_read_counts:
        read_counts = np.array([2])
        reads = np.tile(read, (1, 1, 1))
    else:
        read_counts = None
        reads = np.tile(read, (2, 1, 1))

    expect_genotypes = np.array(
        [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 1, 1], [1, 1, 1, 1]],
        dtype=np.int8,
    )

    # prior probabilities
    expect_probs = np.array(
        [1.999976e-01, 4.499976e-01, 3.000024e-01, 5.000240e-02, 2.000016e-11]
    )

    actual_genotypes, actual_probs = snp_posterior(
        reads[:, 0, :],
        n_alleles=2,
        ploidy=4,
        inbreeding=0.0,
        read_counts=read_counts,
    )

    np.testing.assert_almost_equal(expect_genotypes, actual_genotypes)
    np.testing.assert_almost_equal(expect_probs, actual_probs, decimal=10)
