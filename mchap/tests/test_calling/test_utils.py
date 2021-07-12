import numpy as np
import pytest

from mchap.calling import utils
from mchap.combinatorics import count_unique_genotypes
from mchap.assemble.util import genotype_alleles_as_index


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


def test_posterior_as_array():
    ploidy = 4
    n_haps = 4
    observed_genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 2, 2],
            [0, 2, 2, 2],
            [0, 0, 1, 2],
            [0, 1, 2, 2],
        ]
    )
    observed_probabilities = np.array([0.05, 0.08, 0.22, 0.45, 0.05, 0.15])
    unique_genotypes = count_unique_genotypes(n_haps, ploidy)
    result = utils.posterior_as_array(
        observed_genotypes, observed_probabilities, unique_genotypes
    )
    assert result.sum() == observed_probabilities.sum() == 1
    assert len(result) == unique_genotypes == 35
    for g, p in zip(observed_genotypes, observed_probabilities):
        idx = genotype_alleles_as_index(g)
        assert result[idx] == p
