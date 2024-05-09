import numpy as np
import pytest

from mchap.calling import exact
from mchap import mset
from mchap.testing import simulate_reads
from mchap.assemble.likelihood import log_likelihood
from mchap.assemble.prior import log_genotype_prior
from mchap.jitutils import (
    genotype_alleles_as_index,
    normalise_log_probs,
    index_as_genotype_alleles,
)


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
    actual = exact.genotype_likelihoods(reads, ploidy, haplotypes)
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
    log_likelihoods = exact.genotype_likelihoods(reads, ploidy, haplotypes)
    inbreeding = 0.3
    n_alleles = len(haplotypes)
    expect = np.empty(len(genotypes), dtype=np.float32)
    for i in range(len(expect)):
        genotype = genotypes[i]
        _, dosage = np.unique(genotype, return_counts=True)
        log_prior = log_genotype_prior(
            dosage, log_unique_haplotypes=np.log(n_alleles), inbreeding=inbreeding
        )
        expect[i] = log_likelihoods[i] + log_prior
    expect = normalise_log_probs(expect)
    actual = exact.genotype_posteriors(
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
    actual_genotypes, actual_probs = exact.alternate_dosage_posteriors(
        genotype,
        probabilities,
    )
    np.testing.assert_array_equal(actual_genotypes, expect_genotypes)
    np.testing.assert_array_equal(actual_probs, expect_probs)


def test_call_posterior_mode():

    ploidy = 4
    inbreeding = 0.01
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1],
        ]
    )
    genotype = np.array([0, 0, 0, 2])
    idx_0 = genotype_alleles_as_index(genotype)
    idx_1 = genotype_alleles_as_index(np.array([0, 0, 2, 2]))
    idx_2 = genotype_alleles_as_index(np.array([0, 2, 2, 2]))

    reads = simulate_reads(
        haplotypes[genotype],
        qual=(10, 10),
        uniform_sample=True,
        errors=False,
        n_reads=8,
        error_rate=0,
    )
    reads, counts = mset.unique_counts(reads)

    llks = exact.genotype_likelihoods(reads, ploidy, haplotypes, read_counts=counts)
    probs = exact.genotype_posteriors(
        llks, ploidy, len(haplotypes), inbreeding=inbreeding
    )
    _, phen_probs = exact.alternate_dosage_posteriors(genotype, probs)

    (
        mode_genotype,
        mode_llk,
        mode_genotype_prob,
        mode_phenotype_prob,
    ) = exact.posterior_mode(
        reads,
        4,
        haplotypes,
        read_counts=counts,
        inbreeding=inbreeding,
        return_phenotype_prob=True,
    )

    np.testing.assert_array_equal(genotype, mode_genotype)
    np.testing.assert_almost_equal(llks[idx_0], mode_llk, 5)
    np.testing.assert_almost_equal(probs[idx_0], mode_genotype_prob, 5)
    np.testing.assert_almost_equal(phen_probs.sum(), mode_phenotype_prob, 5)
    np.testing.assert_almost_equal(
        np.sum(probs[[idx_0, idx_1, idx_2]]), mode_phenotype_prob, 5
    )


@pytest.mark.parametrize(
    "ploidy, positions, inbreeding, n_haps, n_reads, frequencies",
    [
        (2, 3, 0.0, 4, 1, None),
        (3, 10, 0.02, 15, 10, True),
        (4, 6, 0.01, 10, 5, False),
        (4, 12, 0.0, 7, 3, None),
        (4, 12, 0.0, 7, 3, True),
        (4, 12, 0.3, 7, 3, None),
        (4, 12, 0.3, 7, 3, True),
        (6, 7, 0.0, 3, 3, True),
        (6, 13, 0.1, 7, 5, True),
    ],
)
def test_call_posterior_mode__fuzz(
    ploidy, positions, inbreeding, n_haps, n_reads, frequencies
):
    np.random.seed(0)

    haplotypes = mset.unique(np.random.randint(0, 2, size=(n_haps, positions)))
    n_haps = len(haplotypes)
    if frequencies is None:
        prior_frequencies = None
    elif frequencies is True:
        prior_frequencies = np.random.rand(n_haps)
        prior_frequencies /= prior_frequencies.sum()
    else:
        prior_frequencies = np.ones(n_haps) / n_haps
    genotype = haplotypes[np.random.randint(0, n_haps, size=ploidy)]
    reads = simulate_reads(genotype, n_reads=n_reads, n_alleles=np.repeat(2, positions))

    # compute full distribution
    llks = exact.genotype_likelihoods(reads=reads, ploidy=ploidy, haplotypes=haplotypes)
    post = exact.genotype_posteriors(
        llks,
        ploidy=ploidy,
        n_alleles=n_haps,
        inbreeding=inbreeding,
        frequencies=prior_frequencies,
    )
    freqs, counts, occur = exact.posterior_allele_frequencies(
        post, ploidy=ploidy, n_alleles=n_haps
    )
    mode = np.argmax(post)
    alleles = index_as_genotype_alleles(mode, ploidy)

    # compute mode using low memory function
    mode_alleles, mode_llk, mode_post, mean_freqs, mean_occur = exact.posterior_mode(
        reads=reads,
        ploidy=ploidy,
        haplotypes=haplotypes,
        inbreeding=inbreeding,
        return_phenotype_prob=False,
        return_posterior_frequencies=True,
        return_posterior_occurrence=True,
        frequencies=prior_frequencies,
    )

    np.testing.assert_array_equal(alleles, mode_alleles)
    np.testing.assert_almost_equal(llks[mode], mode_llk, 5)
    np.testing.assert_almost_equal(post[mode], mode_post, 5)
    np.testing.assert_array_almost_equal(freqs, mean_freqs, 5)
    np.testing.assert_array_almost_equal(counts, mean_freqs * ploidy, 5)
    np.testing.assert_array_almost_equal(occur, mean_occur, 5)


@pytest.mark.parametrize(
    "ploidy, positions, inbreeding, n_haps, n_reads",
    [
        (2, 3, 0.0, 4, 1),
        (3, 10, 0.02, 15, 10),
        (4, 6, 0.01, 10, 5),
        (4, 12, 0.0, 7, 3),
        (4, 12, 0.3, 7, 3),
    ],
)
def test_call_posterior_mode__flat_frequencies(
    ploidy, positions, inbreeding, n_haps, n_reads
):
    np.random.seed(0)

    haplotypes = mset.unique(np.random.randint(0, 2, size=(n_haps, positions)))
    n_haps = len(haplotypes)
    genotype = haplotypes[np.random.randint(0, n_haps, size=ploidy)]
    reads = simulate_reads(genotype, n_reads=n_reads, n_alleles=np.repeat(2, positions))

    (
        mode_alleles_x,
        mode_llk_x,
        mode_post_x,
        mean_freqs_x,
        occur_x,
    ) = exact.posterior_mode(
        reads=reads,
        ploidy=ploidy,
        haplotypes=haplotypes,
        inbreeding=inbreeding,
        return_phenotype_prob=False,
        return_posterior_frequencies=True,
        return_posterior_occurrence=True,
        frequencies=None,
    )

    (
        mode_alleles_y,
        mode_llk_y,
        mode_post_y,
        mean_freqs_y,
        occur_y,
    ) = exact.posterior_mode(
        reads=reads,
        ploidy=ploidy,
        haplotypes=haplotypes,
        inbreeding=inbreeding,
        return_phenotype_prob=False,
        return_posterior_frequencies=True,
        return_posterior_occurrence=True,
        frequencies=np.ones(n_haps) / n_haps,
    )

    np.testing.assert_array_equal(mode_alleles_x, mode_alleles_y)
    np.testing.assert_almost_equal(mode_llk_x, mode_llk_y)
    np.testing.assert_almost_equal(mode_post_x, mode_post_y)
    np.testing.assert_array_almost_equal(mean_freqs_x, mean_freqs_y)
    np.testing.assert_array_almost_equal(occur_x, occur_y)
