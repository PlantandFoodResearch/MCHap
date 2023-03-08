import numpy as np
import pytest
from itertools import combinations_with_replacement

from mchap.assemble.prior import log_dirichlet_multinomial_pmf, log_genotype_prior


@pytest.mark.parametrize(
    "dosage,dispersion,unique_haplotypes,probability",
    [
        [[1, 1, 1, 1], 0.1, 4, 0.0005252100840336145],
        [[2, 1, 1, 0], 0.1, 4, 0.002888655462184875],
        [[2, 2, 0, 0], 0.1, 4, 0.015887605042016827],
        [[3, 1, 0, 0], 0.1, 4, 0.02022058823529413],
        [[4, 0, 0, 0], 0.1, 4, 0.1567095588235296],
        [[1, 1, 1, 1], 0.1, 6, 0.0002670940170940169],
        [[2, 1, 1, 0], 0.1, 6, 0.0014690170940170905],
        [[4, 0, 0, 0], 0.1, 6, 0.07969417735042751],
        [[1, 1, 1, 1], 1.0, 6, 0.007936507936507929],
        [[2, 1, 1, 0], 1.0, 6, 0.007936507936507929],
        [[4, 0, 0, 0], 1.0, 6, 0.007936507936507929],
        [[1, 1, 1, 1], 1.0, 16, 0.0002579979360165113],
        [[2, 1, 1, 0], 1.0, 16, 0.0002579979360165113],
        [[4, 0, 0, 0], 1.0, 16, 0.0002579979360165113],
        [[1, 1, 1, 1], 0.1, 16, 3.483835005574126e-05],
        [[2, 1, 1, 0], 0.1, 16, 0.00019161092530657658],
        [[2, 2, 0, 0], 0.1, 16, 0.0010538600891861704],
        [[3, 1, 0, 0], 0.1, 16, 0.0013412764771460362],
        [[4, 0, 0, 0], 0.1, 16, 0.010394892697881882],
    ],
)
def test_log_dirichlet_multinomial_pmf(
    dosage, dispersion, unique_haplotypes, probability
):
    # Actual probabilities calculated independently using tensorflow-probabilities
    expect = np.log(probability)
    dosage = np.array(dosage)
    actual = log_dirichlet_multinomial_pmf(
        dosage=dosage,
        log_dispersion=np.log(dispersion),
        log_unique_haplotypes=np.log(unique_haplotypes),
    )
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "dosage,unique_haplotypes,inbreeding,probability",
    [
        [[1, 1, 1, 1], 16, 0.0, 0.0003662109375],
        [[2, 1, 1, 0], 16, 0.0, 0.00018310546875],
        [[2, 2, 0, 0], 16, 0.0, 9.155273437499999e-05],
        [[3, 1, 0, 0], 16, 0.0, 6.103515625000003e-05],
        [[4, 0, 0, 0], 16, 0.0, 1.5258789062500007e-05],
        [
            [1, 1, 1, 1],
            16,
            0.1,
            0.00020224831321022713,
        ],  # with F = 0.1 and u_haps = 16 then dispesion = 0.5625
        [[2, 1, 1, 0], 16, 0.1, 0.00028090043501420423],
        [[2, 2, 0, 0], 16, 0.1, 0.0003901394930752838],
        [[3, 1, 0, 0], 16, 0.1, 0.00042655251242897644],
        [[4, 0, 0, 0], 16, 0.1, 0.0006753748113458795],
    ],
)
def test_log_genotype_prior(dosage, unique_haplotypes, inbreeding, probability):
    # Actual probabilities for inbreeding == 0 calculated independently using scipy multinomial
    # Actual probabilities for inbreeding > 0 calculated independently using tensorflow-probabilities
    expect = np.log(probability)
    dosage = np.array(dosage)
    actual = log_genotype_prior(
        dosage=dosage,
        log_unique_haplotypes=np.log(unique_haplotypes),
        inbreeding=inbreeding,
    )
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,unique_haplotypes,inbreeding",
    [
        [2, 16, 0],
        [2, 16, 0.1],
        [2, 16, 0.5],
        [4, 16, 0],
        [4, 16, 0.15],
        [4, 16, 0.45],
        [4, 32, 0],
        [4, 32, 0.1],
        [6, 16, 0],
        [6, 16, 0.2],
    ],
)
def test_log_genotype_prior__normalised(ploidy, unique_haplotypes, inbreeding):
    # tests that the prior probabilities of all possible genotypes sum to 1
    sum_probs = 0.0
    for genotype in combinations_with_replacement(
        list(range(unique_haplotypes)), ploidy
    ):
        _, dosage = np.unique(genotype, return_counts=True)
        lprob = log_genotype_prior(
            dosage,
            log_unique_haplotypes=np.log(unique_haplotypes),
            inbreeding=inbreeding,
        )
        sum_probs += np.exp(lprob)
    np.testing.assert_almost_equal(sum_probs, 1)
