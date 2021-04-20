import numpy as np

from mchap.assemble import calling

from mchap.testing import simulate_reads
from mchap.assemble.likelihood import log_likelihood, log_genotype_prior
from mchap.assemble.util import normalise_log_probs
from mchap.assemble.classes import PosteriorGenotypeDistribution


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


def test_alternate_dosage_posteriors():
    genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 0, 0, 2],
            [0, 0, 1, 2],  # 6
            [0, 1, 1, 2],  # 7
            [1, 1, 1, 2],
            [0, 0, 2, 2],
            [0, 1, 2, 2],  # 10
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
    genotype = np.array([0, 1, 1, 2])
    probabilities = np.zeros(len(genotypes))
    # alt dosages
    probabilities[6] = 0.1
    probabilities[7] = 0.5
    probabilities[10] = 0.2
    # non-alt dosages
    probabilities[5] = 0.1
    probabilities[11] = 0.1

    expect_genotypes = genotypes[[6, 7, 10]]
    expect_probs = np.array([0.1, 0.5, 0.2])
    actual_genotypes, actual_probs = calling.alternate_dosage_posteriors(
        genotype,
        probabilities,
    )
    np.testing.assert_array_equal(actual_genotypes, expect_genotypes)
    np.testing.assert_array_equal(actual_probs, expect_probs)


def test_call_posterior_haplotypes():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 1, 1],  # 1
            [0, 1, 0, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
            [1, 1, 1, 1, 1, 1],  # 4
        ]
    )

    dist1 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[0, 0, 1, 1]],
                haplotypes[[0, 1, 1, 1]],
                haplotypes[[0, 1, 1, 2]],
                haplotypes[[0, 0, 0, 1]],
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.3, 0.1]),
    )
    dist2 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[0, 0, 2, 2]],
                haplotypes[[0, 2, 2, 2]],
                haplotypes[[2, 2, 2, 3]],
                haplotypes[[2, 2, 2, 4]],  # hap 4 at 10%
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.3, 0.1]),
    )
    dist3 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[0, 2, 2, 2]],
                haplotypes[[2, 2, 2, 2]],
                haplotypes[[2, 2, 2, 3]],  # hap 3 at 20%
                haplotypes[[1, 2, 2, 3]],
            ]
        ),
        probabilities=np.array([0.2, 0.6, 0.1, 0.1]),
    )
    posteriors = [dist1, dist2, dist3]
    actual = calling.call_posterior_haplotypes(posteriors, threshold=0.15)
    expect = haplotypes[[0, 2, 1, 3]]  # ref then ordered
    np.testing.assert_array_equal(actual, expect)


def test_call_posterior_haplotypes__no_ref():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 1, 1],  # 1
            [0, 1, 0, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
            [1, 1, 1, 1, 1, 1],  # 4
        ]
    )

    dist1 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[1, 1, 3, 3]],
                haplotypes[[1, 1, 1, 3]],
                haplotypes[[1, 1, 3, 2]],
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.4]),
    )
    dist2 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[2, 2, 2, 3]],
                haplotypes[[2, 2, 3, 3]],
                haplotypes[[2, 3, 3, 3]],
                haplotypes[[2, 3, 3, 4]],  # hap 4 at 10%
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.3, 0.1]),
    )
    dist3 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[1, 2, 2, 2]],
                haplotypes[[2, 2, 2, 2]],
                haplotypes[[2, 2, 2, 3]],
                haplotypes[[1, 2, 2, 3]],
            ]
        ),
        probabilities=np.array([0.2, 0.6, 0.1, 0.1]),
    )
    posteriors = [dist1, dist2, dist3]
    actual = calling.call_posterior_haplotypes(posteriors, threshold=0.15)
    expect = haplotypes[[0, 2, 3, 1]]  # ref added at front ordered
    np.testing.assert_array_equal(actual, expect)
