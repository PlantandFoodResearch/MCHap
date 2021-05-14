import numpy as np
import pytest
from itertools import combinations_with_replacement

from mchap import mset
from mchap.testing import simulate_reads
from mchap.assemble.likelihood import (
    log_likelihood,
    log_likelihood_structural_change,
    log_likelihood_cached,
    new_log_likelihood_cache,
    _log_dirichlet_multinomial_pmf,
    log_genotype_prior,
)


def reference_likelihood(reads, genotype):
    """Reference implementation of likelihood function."""
    n_reads, n_pos, n_nucl = reads.shape
    ploidy, genotype_n_pos = genotype.shape

    # same number of base/variant positions in reads and haplotypes
    assert genotype_n_pos == n_pos

    # convert genotype to matrix of onehot row-vectors
    genotype_matrix = np.zeros((ploidy, n_pos, n_nucl), dtype=reads.dtype)

    for h in range(ploidy):
        for j in range(n_pos):
            i = genotype[h, j]
            # allele is the index of the 1 in the one-hot vector
            genotype_matrix[h, j, i] = 1

    # expand dimentions to use broadcasting
    reads = reads.reshape((n_reads, 1, n_pos, n_nucl))
    genotype = genotype_matrix.reshape((1, ploidy, n_pos, n_nucl))

    # probability of match for each posible allele at each position
    # for every read-haplotype combination
    probs = reads * genotype

    # probability of match at each position for every read-haplotype combination
    # any nans along this axis indicate that ther is a gap in the read at this position
    probs = np.sum(probs, axis=-1)

    # joint probability of match at all positions for every read-haplotype combination
    # nans caused by read gaps should be ignored
    probs = np.nanprod(probs, axis=-1)

    # probability of a read being generated from a genotype is the mean of
    # probabilities of that read matching each haplotype within the genotype
    # this assumes that the reads are generated in equal probortions from each genotype
    probs = np.mean(probs, axis=-1)

    # the probability of a set of reads being generated from a genotype is the
    # joint probability of all reads within the set being generated from that genotype
    probs = np.prod(probs)

    # the likelihood is P(reads|genotype)
    return probs


def test_log_likelihood():
    reads = np.array(
        [
            [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
            [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]],
            [[0.8, 0.2], [0.8, 0.2], [np.nan, np.nan]],
        ]
    )

    genotype = np.array([[0, 0, 0], [0, 0, 1]], dtype=np.int8)

    query = log_likelihood(reads, genotype)
    answer = np.log(reference_likelihood(reads, genotype))

    assert np.round(query, 10) == np.round(answer, 10)


def test_log_likelihood__read_counts():
    genotype = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.random.seed(42)
    reads = simulate_reads(genotype, n_reads=200, qual=(60, 60))
    reads_unique, read_counts = mset.unique_counts(reads)

    expect = log_likelihood(reads, genotype)
    actual = log_likelihood(reads_unique, genotype, read_counts=read_counts)

    # may be slightly off due to float rounding
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,n_base,n_reps",
    [
        [2, 3, 10],
        [2, 5, 10],
        [4, 3, 10],
        [4, 8, 10],
    ],
)
def test_log_likelihood__fuzz(ploidy, n_base, n_reps):
    np.random.seed(0)
    for _ in range(n_reps):
        # 'real' genotype
        haplotypes = np.random.randint(2, size=(ploidy, n_base))
        reads = simulate_reads(haplotypes)

        # 'proposed' genotype
        proposed = np.random.randint(2, size=(ploidy, n_base))

        actual = log_likelihood(reads, proposed)
        expect = np.log(reference_likelihood(reads, proposed))
        np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,n_base,n_reps",
    [
        [2, 3, 10],
        [2, 5, 10],
        [4, 3, 10],
        [4, 8, 1000],
    ],
)
def test_log_likelihood_cache__fuzz(ploidy, n_base, n_reps):
    np.random.seed(0)
    cache = new_log_likelihood_cache(ploidy, n_base, max_alleles=2)
    haplotypes = np.random.randint(2, size=(ploidy, n_base))
    reads = simulate_reads(haplotypes)
    for _ in range(n_reps):
        # 'proposed' genotype
        proposed = np.random.randint(2, size=(ploidy, n_base))
        expect = log_likelihood(reads, proposed)
        actual_1, cache = log_likelihood_cached(reads, proposed, cache, use_cache=False)
        actual_2, cache = log_likelihood_cached(reads, proposed, cache, use_cache=True)
        assert expect == actual_1
        assert expect == actual_2


@pytest.mark.parametrize(
    "reads, genotype, haplotype_indices, interval, final_genotype",
    [
        pytest.param(
            [
                [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
                [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]],
                [[0.8, 0.2], [0.8, 0.2], [np.nan, np.nan]],
            ],
            [[0, 0, 0], [0, 0, 1]],
            [0, 1],
            None,
            [[0, 0, 0], [0, 0, 1]],
            id="2x-no-change",
        ),
        pytest.param(
            [
                [[0.8, 0.2], [0.8, 0.2], [0.8, 0.2]],
                [[0.8, 0.2], [0.8, 0.2], [0.2, 0.8]],
                [[0.8, 0.2], [0.8, 0.2], [np.nan, np.nan]],
            ],
            [[0, 0, 0], [0, 0, 1]],
            [1, 1],
            None,
            [[0, 0, 1], [0, 0, 1]],
            id="2x-overwrite",
        ),
    ],
)
def test_log_likelihood_structural_change(
    reads, genotype, haplotype_indices, interval, final_genotype
):

    reads = np.array(reads, dtype=float)
    genotype = np.array(genotype, dtype=np.int8)
    haplotype_indices = np.array(haplotype_indices, dtype=int)
    final_genotype = np.array(final_genotype, dtype=np.int8)

    query = log_likelihood_structural_change(
        reads, genotype, haplotype_indices, interval=interval
    )
    answer = log_likelihood(reads, final_genotype)
    reference = np.log(reference_likelihood(reads, final_genotype))

    assert query == answer
    assert np.round(query, 10) == np.round(reference, 10)


def test_log_likelihood_structural_change__read_counts():
    genotype = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.random.seed(42)
    reads = simulate_reads(genotype, n_reads=200, qual=(60, 60))
    reads_unique, read_counts = mset.unique_counts(reads)

    interval = (3, 8)
    indices = np.array([0, 2, 2, 3])

    expect = log_likelihood_structural_change(reads, genotype, indices, interval)
    actual = log_likelihood_structural_change(
        reads_unique, genotype, indices, interval, read_counts=read_counts
    )

    # may be slightly off due to float rounding
    np.testing.assert_almost_equal(actual, expect, decimal=10)


@pytest.mark.parametrize(
    "ploidy,n_base,interval,indicies,n_reps",
    [
        [4, 5, (0, 3), [1, 1, 2, 3], 10],
        [4, 5, (1, 4), [1, 0, 2, 3], 10],
        [4, 8, (0, 3), [1, 2, 2, 3], 10],
        [4, 8, (3, 7), [0, 1, 3, 2], 10],
    ],
)
def test_log_likelihood_structural_change__fuzz(
    ploidy, n_base, interval, indicies, n_reps
):
    np.random.seed(0)
    indicies = np.array(indicies)
    for _ in range(n_reps):
        # 'real' genotype
        haplotypes = np.random.randint(2, size=(ploidy, n_base))
        reads = simulate_reads(haplotypes)

        # 'proposed' genotype
        proposed = np.random.randint(2, size=(ploidy, n_base))

        # log like of proposed change
        actual = log_likelihood_structural_change(reads, proposed, indicies, interval)

        # make change
        proposed[:, interval[0] : interval[1]] = proposed[
            indicies, interval[0] : interval[1]
        ]
        expect = np.log(reference_likelihood(reads, proposed))

        np.testing.assert_almost_equal(actual, expect, decimal=10)


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
    actual = _log_dirichlet_multinomial_pmf(dosage, dispersion, unique_haplotypes)
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
    actual = log_genotype_prior(dosage, unique_haplotypes, inbreeding)
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
        lprob = log_genotype_prior(dosage, unique_haplotypes, inbreeding)
        sum_probs += np.exp(lprob)
    np.testing.assert_almost_equal(sum_probs, 1)
