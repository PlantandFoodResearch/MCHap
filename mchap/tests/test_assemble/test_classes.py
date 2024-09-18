import pytest
import numpy as np

from mchap import mset
from mchap.assemble import classes


def test_PosteriorGenotypeDistribution():

    genotypes = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]],
        ],
        dtype=np.int8,
    )

    probabilities = np.array([0.1, 0.6, 0.2, 0.1])

    dist = classes.PosteriorGenotypeDistribution(genotypes, probabilities)

    # mode genotype is one with highest probability
    expect_gen, expect_prob = genotypes[1], probabilities[1]
    actual_gen, actual_prob = dist.mode()
    np.testing.assert_array_equal(expect_gen, actual_gen)
    assert expect_prob, actual_prob

    # support is combination of genotypes with same haplotypes
    expect_phen, expect_probs = genotypes[[1, 2]], probabilities[[1, 2]]
    support = dist.mode_genotype_support()
    actual_phen, actual_probs = support.genotypes, support.probabilities
    np.testing.assert_array_equal(expect_phen, actual_phen)
    np.testing.assert_array_equal(expect_probs, actual_probs)


def test_GenotypeMultiTrace():

    genotypes = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 1, 1]],
        ],
        dtype=np.int8,
    )

    counts = np.array([60, 20, 15, 5])

    single_chain = mset.repeat(genotypes, counts)

    n_chains = 2
    n_steps = len(single_chain)

    genotype_trace = np.tile(single_chain, (n_chains, 1, 1, 1))
    for i in range(n_chains):
        genotype_trace[i] = genotype_trace[i][np.random.permutation(n_steps)]

    llks = np.zeros((n_chains, n_steps))

    trace = classes.GenotypeMultiTrace(genotype_trace, llks)

    burnt = trace.burn(10)
    assert burnt.genotypes.shape == (2, 90, 4, 3)
    assert burnt.llks.shape == (2, 90)

    # posterior sorted from most to least probable
    posterior = trace.posterior()
    np.testing.assert_array_equal(posterior.genotypes, genotypes)
    np.testing.assert_array_equal(posterior.probabilities, counts / counts.sum())


@pytest.mark.parametrize("threshold,expect", [(0.99, 0), (0.8, 0), (0.6, 1)])
def test_GenotypeMultiTrace__replicate_incongruence(threshold, expect):
    haplotypes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    g0 = haplotypes[[0, 0, 1, 2]]  # support 1
    g1 = haplotypes[[0, 1, 1, 2]]  # support 1
    g2 = haplotypes[[0, 1, 2, 2]]  # support 1
    g3 = haplotypes[[0, 0, 2, 2]]  # support 2
    genotypes = np.array([g0, g1, g2, g3])

    t0 = genotypes[[0, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 10:0
    t1 = genotypes[[3, 2, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t2 = genotypes[[0, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 10:0
    t3 = genotypes[[3, 3, 3, 3, 3, 3, 3, 2, 1, 2]]  # 3:7
    trace = classes.GenotypeMultiTrace(
        genotypes=np.array([t0, t1, t2, t3]), llks=np.ones((4, 10))
    )

    actual = trace.replicate_incongruence(threshold)
    assert actual == expect


@pytest.mark.parametrize("threshold,expect", [(0.99, 0), (0.8, 0), (0.6, 2)])
def test_GenotypeMultiTrace__replicate_incongruence__cnv(threshold, expect):
    haplotypes = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
        ]
    )

    g0 = haplotypes[[0, 0, 1, 2]]  # support 1
    g1 = haplotypes[[0, 1, 1, 2]]  # support 1
    g2 = haplotypes[[0, 1, 2, 2]]  # support 1
    g3 = haplotypes[[0, 0, 2, 3]]  # support 2
    g4 = haplotypes[[0, 2, 3, 4]]  # support 3
    genotypes = np.array([g0, g1, g2, g3, g4])

    t0 = genotypes[[3, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t1 = genotypes[[3, 2, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t2 = genotypes[[0, 3, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t3 = genotypes[[3, 3, 4, 4, 4, 3, 4, 4, 4, 4]]  # 3:7
    trace = classes.GenotypeMultiTrace(
        genotypes=np.array([t0, t1, t2, t3]), llks=np.ones((4, 10))
    )
    actual = trace.replicate_incongruence(threshold)
    assert actual == expect


def test_GenotypeSupportDistribution___mode_genotype():
    array = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ],
        dtype=np.int8,
    )
    probs = np.array([0.65, 0.2, 0.1])
    expect = (array[0], probs[0])
    dist = classes.GenotypeSupportDistribution(array, probs)
    actual = dist.mode_genotype()
    np.testing.assert_array_equal(expect[0], actual[0])
    assert expect[1] == actual[1]


def test_PosteriorGenotypeDistribution__allele_frequencies():
    array = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [1, 0, 1], [1, 1, 1]],
            [[0, 0, 0], [1, 0, 1], [0, 1, 0], [1, 1, 1]],
        ],
        dtype=np.int8,
    )
    probs = np.array([0.05, 0.65, 0.2, 0.1])
    dist = classes.PosteriorGenotypeDistribution(array, probs)
    expect_haps = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [1, 0, 1],
            [0, 1, 0],
        ]
    )
    expect_freqs = np.array(
        [
            0.05 * 1 + 0.65 * (3 / 4) + 0.2 * (2 / 4) + 0.1 * (1 / 4),
            0.65 * (1 / 4) + 0.2 * (1 / 4) + 0.1 * (1 / 4),
            0.2 * (1 / 4) + 0.1 * (1 / 4),
            0.1 * (1 / 4),
        ]
    )
    expect_occur = np.array(
        [
            0.05 + 0.65 + 0.2 + 0.1,
            0.65 + 0.2 + 0.1,
            0.2 + 0.1,
            0.1,
        ]
    )
    actual_haps, actual_freqs, actual_occur = dist.allele_frequencies()
    np.testing.assert_array_equal(actual_haps, expect_haps)
    np.testing.assert_array_almost_equal(actual_freqs, expect_freqs)
    np.testing.assert_array_almost_equal(expect_occur, actual_occur)


@pytest.mark.parametrize(
    "threshold,expect",
    [
        pytest.param(
            0.99,
            (
                np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [-1, -1, -1],
                        [-1, -1, -1],
                    ]
                ),
                0.95,
            ),
            id="99",
        ),
        pytest.param(
            0.9,
            (
                np.array(
                    [
                        [0, 0, 0],
                        [1, 1, 1],
                        [-1, -1, -1],
                        [-1, -1, -1],
                    ]
                ),
                0.95,
            ),
            id="90",
        ),
        pytest.param(
            0.8,
            (
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                        [-1, -1, -1],
                    ]
                ),
                0.85,
            ),
            id="85",
        ),
        pytest.param(
            0.6,
            (
                np.array(
                    [
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0],
                        [1, 1, 1],
                    ]
                ),
                0.65,
            ),
            id="65",
        ),
    ],
)
def test_GenotypeSupportDistribution___call_genotype_support(threshold, expect):
    array = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 1, 1]],
            [[0, 0, 0], [0, 0, 0], [1, 1, 1], [1, 1, 1]],
            [[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ],
        dtype=np.int8,
    )
    probs = np.array([0.65, 0.2, 0.1])
    dist = classes.GenotypeSupportDistribution(array, probs)
    actual = dist.call_genotype_support(threshold=threshold)
    np.testing.assert_array_equal(expect[0], actual[0])
    np.testing.assert_almost_equal(expect[1], actual[1])
