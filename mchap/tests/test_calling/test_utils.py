import numpy as np
import pytest

from mchap.calling import utils


@pytest.mark.parametrize(
    "genotype,expect",
    [
        ([0, 1], [1, 1]),
        ([1, 1], [2, 0]),
        ([0, 0, 0, 0], [4, 0, 0, 0]),
        ([0, 1, 1, 2], [1, 2, 0, 1]),
        ([3, 1, 3, 1], [2, 2, 0, 0]),
    ],
)
def test_allelic_dosage(genotype, expect):
    genotype = np.array(genotype)
    actual = utils.allelic_dosage(genotype)
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize(
    "genotype,allele,expect",
    [
        ([0, 1], 0, 1),
        ([1, 1], 0, 0),
        ([1, 1], 1, 2),
        ([1, 1], 2, 0),
        ([0, 0, 0, 0], 3, 0),
        ([0, 1, 1, 2], 2, 1),
        ([3, 1, 3, 1], 3, 2),
    ],
)
def test_count_allele(genotype, allele, expect):
    genotype = np.array(genotype)
    assert expect == utils.count_allele(genotype, allele)
