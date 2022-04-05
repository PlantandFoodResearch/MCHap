import numpy as np
import pytest
from itertools import combinations_with_replacement

from mchap import combinatorics


def test_count_unique_haplotypes():

    u_alleles = [2, 2, 2, 3, 2, 4]

    # the number of unique haplotypes is the product of
    # the number of unique alleles at each variable position within
    # the haplotype
    answer = np.prod(u_alleles)
    query = combinatorics.count_unique_haplotypes(u_alleles)
    assert query == answer


@pytest.mark.parametrize(
    "u_haps,ploidy,answer",
    [
        pytest.param(10, 2, None),
        pytest.param(333, 3, None),
        pytest.param(12, 4, None),
        pytest.param(12, 4, 1365),
        pytest.param(1024, 2, None),  # diploid with 10 biallelic SNPs
        pytest.param(1024, 2, 524800),  # diploid with 10 biallelic SNPs
        pytest.param(1024, 4, 46081900800),  # tetraploid with 10 biallelic SNPs
        pytest.param(1024, 6, 1624866254968320),  # hexaploid with 10 biallelic SNPs
    ],
)
def test_count_unique_genotypes(u_haps, ploidy, answer):

    query = combinatorics.count_unique_genotypes(u_haps, ploidy)

    if answer is None:
        # calculate the answer with itertools

        # a genotype is a multi-set of haplotypes of size ploidy
        # hence the number of multisets can be (inefficiantly) counted
        # by iterating of all ploidy sized combinations of haplotypes with
        # replacement
        answer = 0
        for _ in combinations_with_replacement(np.arange(u_haps), ploidy):
            answer += 1

    assert query == answer


def test_count_unique_genotype_permutations():

    # this is simply the number of unique haplotypes to the power of ploidy

    u_haps = 1024  # 10 biallelic SNPs
    ploidy = 4

    answer = u_haps**ploidy
    query = combinatorics.count_unique_genotype_permutations(u_haps, ploidy)

    assert query == answer


@pytest.mark.parametrize(
    "dosage,answer",
    [
        pytest.param([2, 0], 1),
        pytest.param([1, 1], 2),
        pytest.param([0, 2], 1),
        pytest.param([1, 1, 1], 6),
        pytest.param([2, 1, 0], 3),
        pytest.param([3, 0, 0], 1),
        pytest.param([4, 0, 0, 0], 1),
        pytest.param([2, 2, 0, 0], 6),
        pytest.param([1, 1, 1, 1], 24),
    ],
)
def test_count_genotype_permutations(dosage, answer):
    dosage = np.array(dosage, dtype=int)
    query = combinatorics.count_genotype_permutations(dosage)
    assert query == answer
