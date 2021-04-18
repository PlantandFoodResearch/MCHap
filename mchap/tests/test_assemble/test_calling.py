import numpy as np

from mchap.assemble import calling

from mchap.testing import simulate_reads
from mchap.assemble.likelihood import log_likelihood, log_genotype_prior
from mchap.assemble.util import normalise_log_probs


def test_genotype_likelihoods():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 1, 0, 1, 1, 1],  # 1
            [1, 1, 1, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
        ]
    )
    genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 1, 1, 2],  # true genotype
            [1, 1, 1, 2],
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 2, 2, 2],
            [1, 2, 2, 2],
            [2, 2, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 3],
            [0, 1, 1, 3],
            [1, 1, 1, 3],
            [0, 0, 2, 3],
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [0, 2, 2, 3],
            [1, 2, 2, 3],
            [2, 2, 2, 3],
            [0, 0, 3, 3],
            [0, 1, 3, 3],
            [1, 1, 3, 3],
            [0, 2, 3, 3],
            [1, 2, 3, 3],
            [2, 2, 3, 3],
            [0, 3, 3, 3],
            [1, 3, 3, 3],
            [2, 3, 3, 3],
            [3, 3, 3, 3],
        ]
    )
    correct_genotype = genotypes[7]
    ploidy = len(correct_genotype)
    genotype_haps = haplotypes[correct_genotype]
    error_haps = haplotypes[[3]]
    reads_correct = simulate_reads(
        genotype_haps,
        n_reads=16,
        uniform_sample=True,
        errors=False,
        qual=(60, 60),
    )
    reads_error = simulate_reads(
        error_haps,
        n_reads=1,
        uniform_sample=True,
        errors=False,
        qual=(60, 60),
    )
    _, n_pos, n_nucl = reads_correct.shape
    reads = np.concatenate([reads_correct, reads_error]).reshape(-1, n_pos, n_nucl)
    expect = np.empty(len(genotypes), dtype=np.float32)
    for i in range(len(expect)):
        expect[i] = log_likelihood(reads, haplotypes[genotypes[i]])
    actual = calling.genotype_likelihoods(reads, ploidy, haplotypes)
    np.testing.assert_almost_equal(expect, actual)


def test_genotype_posteriors():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 1, 0, 1, 1, 1],  # 1
            [1, 1, 1, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
        ]
    )
    genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 2],
            [0, 0, 1, 2],
            [0, 1, 1, 2],  # true genotype
            [1, 1, 1, 2],
            [0, 0, 2, 2],
            [0, 1, 2, 2],
            [1, 1, 2, 2],
            [0, 2, 2, 2],
            [1, 2, 2, 2],
            [2, 2, 2, 2],
            [0, 0, 0, 3],
            [0, 0, 1, 3],
            [0, 1, 1, 3],
            [1, 1, 1, 3],
            [0, 0, 2, 3],
            [0, 1, 2, 3],
            [1, 1, 2, 3],
            [0, 2, 2, 3],
            [1, 2, 2, 3],
            [2, 2, 2, 3],
            [0, 0, 3, 3],
            [0, 1, 3, 3],
            [1, 1, 3, 3],
            [0, 2, 3, 3],
            [1, 2, 3, 3],
            [2, 2, 3, 3],
            [0, 3, 3, 3],
            [1, 3, 3, 3],
            [2, 3, 3, 3],
            [3, 3, 3, 3],
        ]
    )
    correct_genotype = genotypes[7]
    ploidy = len(correct_genotype)
    genotype_haps = haplotypes[correct_genotype]
    reads = simulate_reads(
        genotype_haps,
        n_reads=16,
        uniform_sample=True,
        errors=False,
        qual=(60, 60),
    )
    log_likelihoods = calling.genotype_likelihoods(reads, ploidy, haplotypes)
    inbreeding = 0.3
    n_alleles = len(haplotypes)
    expect = np.empty(len(genotypes), dtype=np.float32)
    for i in range(len(expect)):
        genotype = genotypes[i]
        _, dosage = np.unique(genotype, return_counts=True)
        log_prior = log_genotype_prior(dosage, n_alleles, inbreeding=inbreeding)
        expect[i] = log_likelihoods[i] + log_prior
    expect = normalise_log_probs(expect)
    actual = calling.genotype_posteriors(
        log_likelihoods, ploidy, n_alleles, inbreeding=inbreeding
    )
    np.testing.assert_almost_equal(expect, actual)
