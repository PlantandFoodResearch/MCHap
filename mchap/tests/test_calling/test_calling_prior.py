import numpy as np
import pytest

from mchap.calling.prior import (
    inbreeding_as_dispersion,
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
