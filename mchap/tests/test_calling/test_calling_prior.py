import numpy as np
import pytest

from mchap.calling.prior import (
    calculate_alphas,
    log_genotype_allele_prior,
    log_genotype_prior,
)
from mchap.calling.utils import allelic_dosage
from mchap.jitutils import (
    increment_genotype,
    comb_with_replacement,
    genotype_alleles_as_index,
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


@pytest.mark.parametrize("use_frequencies", [False, True])
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
    genotype, variable_allele, unique_haplotypes, inbreeding, use_frequencies, expect
):
    genotype = np.array(genotype)
    if use_frequencies:
        frequencies = np.ones(unique_haplotypes) / unique_haplotypes
    else:
        frequencies = None
    actual = log_genotype_allele_prior(
        genotype,
        variable_allele,
        unique_haplotypes,
        inbreeding,
        frequencies=frequencies,
    )
    np.testing.assert_almost_equal(actual, expect)


@pytest.mark.parametrize("use_frequencies", [False, True])
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
    genotype, unique_haplotypes, inbreeding, use_frequencies, probability
):
    # Actual probabilities for inbreeding == 0 calculated independently using scipy multinomial
    # Actual probabilities for inbreeding > 0 calculated independently using tensorflow-probabilities
    expect = np.log(probability)
    genotype = np.array(genotype)
    if use_frequencies:
        frequencies = np.ones(unique_haplotypes) / unique_haplotypes
    else:
        frequencies = None
    actual = log_genotype_prior(
        genotype, unique_haplotypes, inbreeding, frequencies=frequencies
    )
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize("ploidy", [2, 3, 4, 5])
@pytest.mark.parametrize("inbreeding", [0.0, 0.1, 0.7])
@pytest.mark.parametrize(
    "frequencies",
    [
        [0.5, 0.5],
        [1 / 3, 1 / 3, 1 / 3],
        [0.25, 0.5, 0.125, 0.125],
        [0.0, 0.7, 0.05, 0.1, 0.14, 0.01],
    ],
)
def test_log_genotype_prior__expected_homozygosity(ploidy, inbreeding, frequencies):
    frequencies = np.array(frequencies)
    np.testing.assert_almost_equal(1, frequencies.sum())

    # expected homozygosity based on inbreeding and population allele frequencies
    incidental_ibs = (frequencies**2).sum()
    expected_hom = inbreeding + (1 - inbreeding) * incidental_ibs

    # calculate homozygosity by enumerating over all possible genotypes and
    # multiplying their homozygosity by their prior probability based on
    # inbreeding and population allele frequencies
    n_alleles = len(frequencies)
    n_genotypes = comb_with_replacement(n_alleles, ploidy)
    enumerated_hom = 0.0
    genotype = np.zeros(ploidy, np.int8)
    for _ in range(n_genotypes):
        dosage = allelic_dosage(genotype)
        af = dosage / ploidy
        ibs = (af**2).sum()
        hom = (ibs * ploidy - 1) / (ploidy - 1)
        lprior = log_genotype_prior(
            genotype,
            n_alleles,
            inbreeding=inbreeding,
            frequencies=frequencies,
        )
        enumerated_hom += np.exp(lprior) * hom
        increment_genotype(genotype)

    np.testing.assert_almost_equal(
        expected_hom,
        enumerated_hom,
    )


@pytest.mark.parametrize("ploidy", [4, 6])
@pytest.mark.parametrize("inbreeding", [0.1])
@pytest.mark.parametrize(
    "frequencies",
    [
        [0.5, 0.5],
        [1 / 3, 1 / 3, 1 / 3],
        [0.25, 0.5, 0.125, 0.125],
        [0.01, 0.69, 0.05, 0.1, 0.14, 0.01],
    ],
)
def test_log_genotype_prior__simulation(ploidy, inbreeding, frequencies):
    frequencies = np.array(frequencies)

    n_alleles = len(frequencies)
    n_genotypes = comb_with_replacement(n_alleles, ploidy)
    exact_priors = np.zeros(n_genotypes)
    genotype = np.zeros(ploidy, np.int8)
    for i in range(n_genotypes):
        lprior = log_genotype_prior(
            genotype,
            n_alleles,
            inbreeding=inbreeding,
            frequencies=frequencies,
        )
        exact_priors[i] = np.exp(lprior)
        increment_genotype(genotype)

    simulated_priors = np.zeros(n_genotypes)
    sims = 5000
    for _ in range(sims):
        alphas = calculate_alphas(inbreeding, frequencies)
        ac = np.random.multinomial(ploidy, np.random.dirichlet(alphas))
        gt = np.repeat(np.arange(len(ac)), ac)
        idx = genotype_alleles_as_index(gt)
        simulated_priors[idx] += 1 / sims

    np.testing.assert_almost_equal(
        exact_priors,
        simulated_priors,
        decimal=2,
    )
