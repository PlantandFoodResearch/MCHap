import numpy as np
import pytest

from mchap.calling.prior import (
    inbreeding_as_dispersion,
    log_dirichlet_multinomial_pmf,
    log_genotype_allele_prior,
)


@pytest.mark.parametrize(
    "inbreeding,unique_haplotypes,dispersion",
    [
        [0.1, 4, 2.25],
        [0.5, 4, 0.25],
        [1, 4, 0.0],
        [0.1, 16, 0.5625],
        [0.5, 16, 0.0625],
        [1, 16, 0.0],
        [0.1, 32, 0.28125],
    ],
)
def test_inbreeding_as_dispersion(inbreeding, unique_haplotypes, dispersion):
    actual = inbreeding_as_dispersion(inbreeding, unique_haplotypes)
    assert actual == dispersion


@pytest.mark.parametrize(
    "allele_counts,dispersion,expect",
    [
        # tests for single allele
        [[1, 0, 0, 0], [2.25, 2.25, 2.25, 2.25], np.log(0.25)],  # F = 0.1
        [[0, 1, 0, 0], [2.25, 2.25, 2.25, 2.25], np.log(0.25)],  # F = 0.1
        [[1, 0, 0, 0], [0.25, 0.25, 0.25, 0.25], np.log(0.25)],  # F = 0.5
        # tests for full dosage
        [[2, 0, 0, 0], [2.25, 2.25, 2.25, 2.25], np.log(0.08125)],  # F = 0.1
        [[0, 2, 0, 0], [2.25, 2.25, 2.25, 2.25], np.log(0.08125)],  # F = 0.1
        [[1, 0, 1, 0], [2.25, 2.25, 2.25, 2.25], np.log(0.1125)],  # F = 0.1
        [
            [1, 1, 1, 1],
            [2.25, 2.25, 2.25, 2.25],
            np.log(0.05177556818181821),
        ],  # F = 0.1
        [
            [3, 0, 0, 1],
            [2.25, 2.25, 2.25, 2.25],
            np.log(0.023544034090909046),
        ],  # F = 0.1
        [
            [0, 2, 2, 0],
            [2.25, 2.25, 2.25, 2.25],
            np.log(0.02700639204545451),
        ],  # F = 0.1
        [[0, 2, 0, 0], [0.25, 0.25, 0.25, 0.25], np.log(0.15625)],  # F = 0.5
        [[0, 1, 0, 1], [0.25, 0.25, 0.25, 0.25], np.log(0.0625)],  # F = 0.5
        [[1, 1, 1, 1], [0.25, 0.25, 0.25, 0.25], np.log(0.00390625)],  # F = 0.5
        [[0, 2, 0, 2], [0.25, 0.25, 0.25, 0.25], np.log(0.0244140625)],  # F = 0.5
        [[1, 3, 0, 0], [0.25, 0.25, 0.25, 0.25], np.log(0.029296875)],  # F = 0.5
        [[0, 0, 4, 0], [0.25, 0.25, 0.25, 0.25], np.log(0.09521484375)],  # F = 0.5
        [
            [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.full(16, 0.0625),
            np.log(0.01747703552246084),
        ],  # F = 0.5
        [
            [3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.full(16, 0.0625),
            np.log(0.001426696777343747),
        ],  # F = 0.5
        [
            [2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.full(16, 0.0625),
            np.log(0.0011024475097656263),
        ],  # F = 0.5
        [
            [2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            np.full(16, 0.0625),
            np.log(0.0001296997070312497),
        ],  # F = 0.5
        # test for single allele given other alleles as constants
        [[1, 0, 0, 0], 2.25 + np.array([3, 0, 0, 0]), np.log(0.4375)],  # F = 0.1
        [[0, 1, 0, 0], 2.25 + np.array([3, 0, 0, 0]), np.log(0.1875)],  # F = 0.1
        [
            [0, 1, 0, 0],
            2.25 + np.array([2, 1, 0, 0]),
            np.log(0.2708333333333328),
        ],  # F = 0.1
        [[1, 0, 0, 0], 0.25 + np.array([3, 0, 0, 0]), np.log(0.8125)],  # F = 0.5
        [[0, 1, 0, 0], 0.25 + np.array([3, 0, 0, 0]), np.log(0.0625)],  # F = 0.5
        [[0, 1, 0, 0], 0.25 + np.array([2, 1, 0, 0]), np.log(0.3125)],  # F = 0.5
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0.5625 + np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.log(0.296875),
        ],  # F = 0.1
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0.5625 + np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.log(0.046875),
        ],  # F = 0.1
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0.5625 + np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.log(0.13020833333333315),
        ],  # F = 0.1
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0.0625 + np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.log(0.765625),
        ],  # F = 0.5
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0.0625 + np.array([3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.log(0.015625),
        ],  # F = 0.5
        [
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            0.0625 + np.array([2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.log(0.265625),
        ],  # F = 0.5
    ],
)
def test_log_dirichlet_multinomial_pmf(allele_counts, dispersion, expect):
    """Expected values calculated with tensorflow-probabilities."""
    allele_counts = np.array(allele_counts, int)
    dispersion = np.array(dispersion, float)
    actual = log_dirichlet_multinomial_pmf(allele_counts, dispersion)
    np.testing.assert_almost_equal(actual, expect)


@pytest.mark.parametrize(
    "genotype,variable_allele,unique_haplotypes,inbreeding,expect",
    [
        [[0, 0, 0, 0], 0, 4, 0.0, np.log(1 / 4)],
        [[0, 1, 2, 3], 0, 4, 0.0, np.log(1 / 4)],
        [[0, 0, 0, 0], 0, 4, 0.1, np.log(0.4375)],
        [[0, 0, 0, 0], 1, 4, 0.1, np.log(0.4375)],
        [[0, 0, 0, 0], 0, 4, 0.5, np.log(0.8125)],
        [[0, 0, 0, 0], 1, 4, 0.5, np.log(0.8125)],
        [[0, 0, 0, 1], 3, 4, 0.5, np.log(0.0625)],
        [[2, 2, 3, 2], 2, 4, 0.5, np.log(0.0625)],
        [[0, 0, 3, 3], 3, 4, 0.5, np.log(0.3125)],
        [[0, 0, 0, 0], 0, 16, 0.0, np.log(1 / 16)],
        [[0, 1, 2, 3], 0, 16, 0.0, np.log(1 / 16)],
        [[0, 0, 3, 3], 3, 16, 0.1, np.log(0.13020833333333315)],
        [[0, 3, 3, 3], 0, 16, 0.5, np.log(0.015625)],
    ],
)
def test_log_genotype_allele_prior(
    genotype, variable_allele, unique_haplotypes, inbreeding, expect
):
    genotype = np.array(genotype)
    actual = log_genotype_allele_prior(
        genotype, variable_allele, unique_haplotypes, inbreeding
    )
    np.testing.assert_almost_equal(actual, expect)
