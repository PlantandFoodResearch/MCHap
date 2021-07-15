import numpy as np
import pytest
from numba import njit

from mchap.calling.likelihood import (
    log_likelihood_alleles,
    log_likelihood_alleles_cached,
)
from mchap.assemble.likelihood import log_likelihood
from mchap.testing import simulate_reads


@pytest.mark.parametrize(
    "seed",
    [11, 42, 13, 0, 12234, 213, 45436, 1312, 374645],
)
def test_log_likelihood_alleles(seed):
    np.random.seed(seed)

    n_pos = np.random.randint(3, 13)
    n_haps = np.random.randint(2, 20)
    n_reads = np.random.randint(2, 15)
    ploidy = np.random.randint(2, 9)
    haplotypes = np.random.randint(0, 2, size=(n_haps, n_pos))

    for _ in range(20):
        alleles = np.random.randint(0, n_haps, size=ploidy)
        alleles.sort()
        genotype = haplotypes[alleles]
        reads = simulate_reads(
            genotype,
            n_alleles=np.full(n_pos, 2, int),
            n_reads=n_reads,
        )
        read_counts = np.random.randint(1, 10, size=n_reads)

        expect = log_likelihood(reads=reads, read_counts=read_counts, genotype=genotype)
        actual = log_likelihood_alleles(
            reads=reads,
            read_counts=read_counts,
            haplotypes=haplotypes,
            genotype_alleles=alleles,
        )
        assert expect == actual


@njit(cache=True)
def _compare_cached_none_cached_likelihoods(reads, read_counts, haplotypes, genotypes):
    cache = {}
    cache[-1] = np.nan

    n_gens = len(genotypes)
    expect = np.empty(n_gens)
    actual = np.empty(n_gens)

    for i in range(n_gens):
        expect[i] = log_likelihood_alleles(
            reads=reads,
            read_counts=read_counts,
            haplotypes=haplotypes,
            genotype_alleles=genotypes[i],
        )
        actual[i] = log_likelihood_alleles_cached(
            reads=reads,
            read_counts=read_counts,
            haplotypes=haplotypes,
            genotype_alleles=genotypes[i],
            cache=cache,
        )
    return expect, actual


@pytest.mark.parametrize(
    "seed",
    [11, 42, 13, 0, 12234, 213, 45436, 1312, 374645],
)
def test_log_likelihood_alleles_cached(seed):
    np.random.seed(seed)

    n_pos = np.random.randint(3, 13)
    n_haps = np.random.randint(2, 9)
    n_reads = np.random.randint(2, 15)
    ploidy = np.random.randint(2, 4)
    haplotypes = np.random.randint(0, 2, size=(n_haps, n_pos))

    true_alleles = np.random.randint(0, n_haps, size=ploidy)
    true_alleles.sort()
    genotype = haplotypes[true_alleles]
    reads = simulate_reads(
        genotype,
        n_alleles=np.full(n_pos, 2, int),
        n_reads=n_reads,
    )
    read_counts = np.random.randint(1, 10, size=n_reads)

    n_tests = 200
    test_genotype_alleles = np.random.randint(0, n_haps, size=(n_tests, ploidy))

    expect, actual = _compare_cached_none_cached_likelihoods(
        reads=reads,
        read_counts=read_counts,
        haplotypes=haplotypes,
        genotypes=test_genotype_alleles,
    )
    np.testing.assert_array_almost_equal(expect, actual)
