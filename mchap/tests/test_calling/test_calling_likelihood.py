import numpy as np
import pytest

from mchap.calling.likelihood import log_likelihood_alleles
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
