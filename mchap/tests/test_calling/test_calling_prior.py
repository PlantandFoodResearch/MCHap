import numpy as np
import pytest

from mchap.calling.prior import (
    calculate_alphas,
    log_genotype_allele_prior,
    log_genotype_prior,
)


@pytest.mark.parametrize(
    "inbreeding,frequencies,alphas",
    [
        [0.1, np.array([0.25, 0.25, 0.25, 0.25]), np.array([2.25, 2.25, 2.25, 2.25])],
        [
            0.1,
            np.array([0.0, 0.25, 0.25, 0.25, 0.25]),
            np.array([0.0, 2.25, 2.25, 2.25, 2.25]),
        ],
        [0.5, np.array([0.25, 0.25, 0.25, 0.25]), np.array([0.25, 0.25, 0.25, 0.25])],
        [
            0.5,
            np.array([0.0, 0.25, 0.25, 0.25, 0.25]),
            np.array([0.0, 0.25, 0.25, 0.25, 0.25]),
        ],
    ],
)
def test_calculate_alphas(inbreeding, frequencies, alphas):
    actual = calculate_alphas(inbreeding, frequencies)
    np.testing.assert_array_almost_equal(actual, alphas)


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
def test_calculate_alphas__flat_frequency(inbreeding, unique_haplotypes, dispersion):
    actual = calculate_alphas(inbreeding, 1 / unique_haplotypes)
    assert actual == dispersion


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
def test_log_genotype_allele_prior__flat_frequency(
    genotype, variable_allele, unique_haplotypes, inbreeding, expect
):
    genotype = np.array(genotype)
    actual = log_genotype_allele_prior(
        genotype, variable_allele, unique_haplotypes, inbreeding
    )
    np.testing.assert_almost_equal(actual, expect)


@pytest.mark.parametrize(
    "genotype,unique_haplotypes,inbreeding,probability",
    [
        [[0, 1, 2, 3], 16, 0.0, 0.0003662109375],
        [[2, 3, 4, 2], 16, 0.0, 0.00018310546875],
        [[1, 2, 1, 2], 16, 0.0, 9.155273437499999e-05],
        [[0, 7, 0, 0], 16, 0.0, 6.103515625000003e-05],
        [[3, 3, 3, 3], 16, 0.0, 1.5258789062500007e-05],
        [
            [0, 1, 2, 3],
            16,
            0.1,
            0.00020224831321022713,
        ],  # with F = 0.1 and u_haps = 16 then dispesion = 0.5625
        [[2, 3, 4, 2], 16, 0.1, 0.00028090043501420423],
        [[1, 2, 1, 2], 16, 0.1, 0.0003901394930752838],
        [[0, 7, 0, 0], 16, 0.1, 0.00042655251242897644],
        [[3, 3, 3, 3], 16, 0.1, 0.0006753748113458795],
    ],
)
def test_log_genotype_prior__flat_frequency(
    genotype, unique_haplotypes, inbreeding, probability
):
    # Actual probabilities for inbreeding == 0 calculated independently using scipy multinomial
    # Actual probabilities for inbreeding > 0 calculated independently using tensorflow-probabilities
    expect = np.log(probability)
    genotype = np.array(genotype)
    actual = log_genotype_prior(genotype, unique_haplotypes, inbreeding)
    np.testing.assert_almost_equal(actual, expect, decimal=10)
