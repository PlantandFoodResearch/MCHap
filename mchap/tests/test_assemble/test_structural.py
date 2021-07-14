import numpy as np
import pytest

from mchap.testing import simulate_reads, metropolis_hastings_transitions
from mchap.encoding import integer
from mchap.jitutils import (
    seed_numba,
    normalise_log_probs,
    get_haplotype_dosage,
    structural_change,
)
from mchap.assemble.likelihood import (
    log_likelihood,
    log_genotype_prior,
    new_log_likelihood_cache,
)
from mchap.assemble import structural
from mchap import mset


@pytest.mark.parametrize(
    "breaks,n",
    [
        pytest.param(0, 1),
        pytest.param(2, 5),
        pytest.param(0, 11),
        pytest.param(1, 11),
        pytest.param(3, 11),
        pytest.param(10, 11),
        pytest.param(20, 100),
    ],
)
def test_random_breaks(breaks, n):

    # intervals are of random length but should be within several constraints
    # so test that they
    intervals = structural.random_breaks(breaks, n)

    # number of intervals produced is 1 + the break points between intervals
    assert len(intervals) == breaks + 1

    # all intervals must be of length 1 or greater
    lengths = intervals[:, 1] - intervals[:, 0]
    assert np.all(lengths > 0)

    # all positions must be contained in one and only one interval
    # i.e. the end position of one interval must also be the start of the next

    # check first interval starts at 0
    assert intervals[0, 0] == 0

    # check last interval ends at end point
    assert intervals[-1, 1] == n

    # check all intervalls are adjacent
    stops = intervals[1:, 0]  # all but first interval
    starts = intervals[:-1, 1]  # all but last interval
    np.testing.assert_array_equal(stops, starts)


@pytest.mark.parametrize(
    "genotype,haplotype_indices,interval,answer",
    [
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            [0, 1, 2, 3],  # keep same haplotypes
            None,
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            id="4x-no-change",
        ),
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            [2, 1, 0, 3],  # switch hap 0 with hap 2 (this is a meaningless change)
            None,
            [
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 2, 1, 0],
            ],
            id="4x-switch",
        ),
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            [0, 1, 0, 3],  # overwrite hap 2 with hap 0
            None,  # full haplotype
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 2, 1, 0],
            ],
            id="4x-overwrite",
        ),
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            [3, 1, 2, 3],  # overwrite hap 0 with hap 3
            (3, 5),  # within this interval
            [
                [0, 1, 0, 2, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            id="4x-partial-overwrite",
        ),
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            [3, 1, 2, 0],  # switch hap 0 with hap 3
            (3, 5),  # within this interval
            [
                [0, 1, 0, 2, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
            ],
            id="4x-recombine",
        ),
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
            ],
            [2, 3, 3, 1],  # madness
            (3, 6),  # within this interval
            [
                [0, 1, 0, 1, 1, 0],
                [0, 1, 1, 2, 1, 0],
                [0, 1, 1, 2, 1, 0],
                [0, 1, 1, 1, 1, 1],
            ],
            id="4x-multi",
        ),
    ],
)
def test_structural_change(genotype, haplotype_indices, interval, answer):

    genotype = np.array(genotype, dtype=np.int8)
    haplotype_indices = np.array(haplotype_indices, dtype=np.int8)
    answer = np.array(answer, dtype=int)

    structural_change(genotype, haplotype_indices, interval=interval)

    np.testing.assert_array_equal(genotype, answer)


@pytest.mark.parametrize(
    "genotype,interval,answer",
    [
        pytest.param([[0, 1, 0], [0, 1, 0]], None, [[0, 0], [0, 0]], id="2x-hom"),
        pytest.param([[0, 1, 0], [0, 1, 1]], None, [[0, 0], [1, 0]], id="2x-het"),
        pytest.param(
            [[0, 1, 0], [0, 1, 1]], (0, 2), [[0, 0], [0, 1]], id="2x-het-hom-interval"
        ),
        pytest.param(
            [[0, 1, 0], [0, 1, 1]], (1, 3), [[0, 0], [1, 0]], id="2x-het-het-interval"
        ),
        pytest.param(
            [[0, 1, 0], [0, 1, 1]],
            (2, 2),
            [[0, 0], [0, 1]],
            id="2x-het-zero-width-interval",
        ),
        pytest.param(
            [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
            None,
            [[0, 0], [1, 0], [0, 0]],
            id="3x-2:1",
        ),
        pytest.param(
            [[0, 1, 0], [0, 1, 1], [0, 1, 1]],
            None,
            [[0, 0], [1, 0], [1, 0]],
            id="3x-1:2",
        ),
        pytest.param(
            [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
            None,
            [[0, 0], [1, 0], [1, 0], [3, 0]],
            id="4x-1:2:1",
        ),
        pytest.param(
            [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
            (0, 3),
            [[0, 0], [1, 0], [1, 0], [0, 3]],
            id="4x-2:2-interval",
        ),
        pytest.param(
            [
                [0, 1, 0, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 0],
            ],
            (2, 5),
            [[0, 0], [1, 0], [1, 2], [1, 2]],
            id="4x-1:3-interval",
        ),
    ],
)
def test_haplotype_segment_labels(genotype, interval, answer):

    genotype = np.array(genotype, dtype=np.int8)
    answer = np.array(answer, dtype=int)

    query = structural.haplotype_segment_labels(genotype, interval=interval)

    np.testing.assert_array_equal(query, answer)


@pytest.mark.parametrize(
    "labels,answer",
    [
        pytest.param(
            [[0, 0], [1, 0]],
            np.empty((0, 2, 2), dtype=np.int8),  # no options
            id="2x-0",
        ),
        pytest.param([[0, 0], [1, 1]], [[[1, 0], [0, 1]]], id="2x-1"),  # 1 option
        pytest.param(
            [[0, 0], [0, 1], [0, 1], [0, 0]],
            np.empty((0, 4, 2), dtype=np.int8),  # no options
            id="4x-0",
        ),
        pytest.param(
            [[0, 0], [0, 1], [2, 1], [0, 0]],
            [[[2, 0], [0, 1], [0, 1], [0, 0]]],  # 1 option
            id="4x-1",
        ),
        pytest.param(
            [[0, 0], [0, 1], [2, 1], [3, 0]],
            [
                [[2, 0], [0, 1], [0, 1], [3, 0]],
                [[0, 0], [3, 1], [2, 1], [0, 0]],
                [[0, 0], [0, 1], [3, 1], [2, 0]],
            ],  # 3 options
            id="4x-3",
        ),
        pytest.param(
            [[0, 0], [1, 1], [2, 2], [3, 3]],
            [
                [[1, 0], [0, 1], [2, 2], [3, 3]],
                [[2, 0], [1, 1], [0, 2], [3, 3]],
                [[3, 0], [1, 1], [2, 2], [0, 3]],
                [[0, 0], [2, 1], [1, 2], [3, 3]],
                [[0, 0], [3, 1], [2, 2], [1, 3]],
                [[0, 0], [1, 1], [3, 2], [2, 3]],
            ],  # all the options
            id="4x-6",
        ),
    ],
)
def test_recombination_step_options(labels, answer):

    labels = np.array(labels)
    answer = np.array(answer)

    n_options = structural.recombination_step_n_options(labels)
    query = structural.recombination_step_options(labels)

    assert len(query) == n_options
    np.testing.assert_array_equal(query, answer)


@pytest.mark.parametrize(
    "labels,answer",
    [
        pytest.param(
            [[0, 0], [0, 0]],
            np.empty((0, 2, 2), dtype=np.int8),  # no options
            id="2x-hom",
        ),
        pytest.param(
            [[0, 0], [1, 0]],
            np.empty((0, 2, 2), dtype=np.int8),  # no options
            id="2x-0",
        ),
        pytest.param(
            [[0, 0], [0, 0], [0, 0], [3, 0]],
            [[[3, 0], [0, 0], [0, 0], [3, 0]]],
            id="4x-1",
        ),
        pytest.param(
            [[0, 0], [0, 1], [2, 0], [2, 0]],
            [
                [[2, 0], [0, 1], [2, 0], [2, 0]],
                [[0, 0], [2, 1], [2, 0], [2, 0]],
                [[0, 0], [0, 1], [0, 0], [2, 0]],
            ],
            id="4x-3",
        ),
        pytest.param(
            [[0, 0], [0, 1], [2, 2], [2, 3], [4, 4], [4, 5]],
            [
                [[2, 0], [0, 1], [2, 2], [2, 3], [4, 4], [4, 5]],
                [[4, 0], [0, 1], [2, 2], [2, 3], [4, 4], [4, 5]],
                [[0, 0], [2, 1], [2, 2], [2, 3], [4, 4], [4, 5]],
                [[0, 0], [4, 1], [2, 2], [2, 3], [4, 4], [4, 5]],
                [[0, 0], [0, 1], [0, 2], [2, 3], [4, 4], [4, 5]],
                [[0, 0], [0, 1], [4, 2], [2, 3], [4, 4], [4, 5]],
                [[0, 0], [0, 1], [2, 2], [0, 3], [4, 4], [4, 5]],
                [[0, 0], [0, 1], [2, 2], [4, 3], [4, 4], [4, 5]],
                [[0, 0], [0, 1], [2, 2], [2, 3], [0, 4], [4, 5]],
                [[0, 0], [0, 1], [2, 2], [2, 3], [2, 4], [4, 5]],
                [[0, 0], [0, 1], [2, 2], [2, 3], [4, 4], [0, 5]],
                [[0, 0], [0, 1], [2, 2], [2, 3], [4, 4], [2, 5]],
            ],
            id="6x",
        ),
    ],
)
def test_dosage_step_options(labels, answer):

    labels = np.array(labels)
    answer = np.array(answer)

    n_options = structural.dosage_step_n_options(labels)
    query = structural.dosage_step_options(labels)

    assert len(query) == n_options
    np.testing.assert_array_equal(query, answer)


@pytest.mark.parametrize(
    "use_cache",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "inbreeding",
    [
        0.0,
        0.25,
    ],
)
def test_interval_step__recombination(use_cache, use_read_counts, inbreeding):
    np.random.seed(42)
    seed_numba(42)

    # true haplotypes
    haplotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 1],
        ]
    )
    ploidy, n_base = haplotypes.shape
    unique_haplotypes = 2 ** n_base

    reads = simulate_reads(
        haplotypes,
        n_reads=4,
        uniform_sample=True,
        errors=False,
        error_rate=0.2,
        qual=(60, 60),
    )

    # interval includes first 2 bases
    interval = (0, 2)

    # All unique re-arrangements reachable by recombination
    # from worst to best match.
    # Not all genotypes can be reached by recombination alone
    # however we still calculate their priors as though all
    # can be reached to be consistent with the prior calculation
    # within the `interval_step` function.
    # By calculating the exact posterior for only the genotypes
    # that can be reached we ensure that the exact and simulated
    # posteriors are equivalent.
    genotypes = np.array(
        [
            # [[0, 2][0, 2][2, 0][2, 1]], 2:1:1
            # worst: no matching haps
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1]],
            # [[0, 1][0, 2][2, 0][2, 2]], 1:1:1:1
            # 2nd worst: 2 matching haps
            [[0, 0, 0, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]],
            # [[0, 0][0, 1][2, 2][2, 2]], 2:1:1
            # 2nd best: 2 matching haps one with 2 copies
            [[0, 0, 0, 0], [0, 0, 0, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            # [[0, 0][0, 2][2, 1][2, 2]], 1:1:1:1
            # best: all matching haps
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 1]],
        ]
    )

    # calculate exact posteriors for reachable genotypes
    log_expect = np.empty(len(genotypes))
    dosage = np.empty(ploidy, int)
    for i, g in enumerate(genotypes):
        get_haplotype_dosage(dosage, g)
        llk = log_likelihood(reads, g)
        lprior = log_genotype_prior(dosage, unique_haplotypes, inbreeding=inbreeding)
        log_expect[i] = llk + lprior
    expect = normalise_log_probs(log_expect)

    # mcmc simulation
    # additional parameters
    if use_cache:
        cache = new_log_likelihood_cache(ploidy, n_base, 2)
    else:
        cache = None
    if use_read_counts:
        reads, read_counts = mset.unique_counts(reads)
    else:
        read_counts = None

    # initial genotype
    genotype = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    llk = log_likelihood(reads, genotype)
    # count choices of each option
    counts = {}
    for g in genotypes:
        counts[g.tobytes()] = 0
    # simulation

    for _ in range(25_000):
        llk, cache = structural.interval_step(
            genotype,
            reads,
            llk,
            unique_haplotypes=unique_haplotypes,
            interval=interval,
            step_type=0,
            inbreeding=inbreeding,
            cache=cache,
            read_counts=read_counts,
        )
        genotype = integer.sort(genotype)
        counts[genotype.tobytes()] += 1
    totals = np.zeros(len(genotypes), dtype=int)
    for i, g in enumerate(genotypes):
        totals[i] = counts[g.tobytes()]

    actual = totals / totals.sum()

    # simulation posteriors should be almost equal
    # but there may be some auto-correlation present
    # in the simulation
    np.testing.assert_array_almost_equal(
        expect,
        actual,
        decimal=2,
    )


@pytest.mark.parametrize(
    "use_cache",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "inbreeding",
    [
        0.0,
        0.25,
    ],
)
def test_interval_step__dosage_swap(use_cache, use_read_counts, inbreeding):
    """Test posterior probabilities of a small dosage swap only
    MCMC compared to posterior probabilites calculated from all
    possible genotypes with and without MH acceptance probability.
    """
    np.random.seed(42)
    seed_numba(42)

    # true haplotypes
    haplotypes = np.array(
        [
            [0, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    ploidy, n_base = haplotypes.shape
    unique_haplotypes = 2 ** n_base

    reads = simulate_reads(
        haplotypes,
        n_reads=4,
        uniform_sample=True,
        errors=False,
        error_rate=0.2,
        qual=(60, 60),
    )

    # interval includes first 2 bases
    interval = (0, 2)

    # All unique re-arrangements reachable by dosage-swap
    # Not all genotypes can be reached by dosage-swap alone
    # however we still calculate their priors as though all
    # can be reached to be consistent with the prior calculation
    # within the `interval_step` function.
    # By calculating the exact posterior for only the genotypes
    # that can be reached we ensure that the exact and simulated
    # posteriors are equivalent.
    genotypes = np.array(
        [
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]],  # 0  2:1:1
            [[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0]],  # 1  2:1:1
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],  # 2  2:2
            [[0, 0, 0, 0], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 1, 1]],  # 3  1:1:1:1
            [[0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0]],  # 4  2:2
            [
                [0, 0, 0, 0],  # 5  2:1:1
                [1, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            [[0, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 1, 1]],  # 6  2:1:1
        ]
    )

    # calculate exact posteriors for reachable genotypes
    llks = np.empty(len(genotypes))
    lpriors = np.empty(len(genotypes))
    dosage = np.empty(ploidy, int)
    for i, g in enumerate(genotypes):
        get_haplotype_dosage(dosage, g)
        llks[i] = log_likelihood(reads, g)
        lpriors[i] = log_genotype_prior(
            dosage, unique_haplotypes, inbreeding=inbreeding
        )
    exact_posteriors = normalise_log_probs(llks + lpriors)

    # now calculate posteriors using a full transition matrix
    # of legal dosage swap transitions between genotypes
    legal_transitions = np.array(
        [
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 0, 0],
        ]
    )
    # MH-transiton probs
    mh_probs = metropolis_hastings_transitions(legal_transitions, llks, np.exp(lpriors))
    # posterior probs based on long run behavior
    matrix_posteriors = np.linalg.matrix_power(mh_probs, 1000)[0]

    # mcmc simulation
    # additional parameters
    if use_cache:
        cache = new_log_likelihood_cache(ploidy, n_base, 2)
    else:
        cache = None
    if use_read_counts:
        reads, read_counts = mset.unique_counts(reads)
    else:
        read_counts = None

    # initial genotype for simulation
    genotype = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]
    )
    llk = log_likelihood(reads, genotype)

    # mcmc simulation
    counts = {}
    for g in genotypes:
        counts[g.tobytes()] = 0
    for i in range(25_000):
        llk, cache = structural.interval_step(
            genotype,
            reads,
            llk,
            unique_haplotypes=unique_haplotypes,
            interval=interval,
            step_type=1,
            inbreeding=inbreeding,
            read_counts=read_counts,
            cache=cache,
        )
        genotype = integer.sort(genotype)
        counts[genotype.tobytes()] += 1

    # posteriors from simulation
    totals = np.zeros(len(genotypes), dtype=int)
    for i, g in enumerate(genotypes):
        totals[i] = counts[g.tobytes()]
    simulation_posteriors = totals / totals.sum()

    # exact and matrix posteriors should be arbitrarily close
    np.testing.assert_array_almost_equal(exact_posteriors, matrix_posteriors, decimal=6)

    # simulation posteriors should be almost equal
    # but there may be some auto-correlation present
    # in the simulation
    np.testing.assert_array_almost_equal(
        exact_posteriors,
        simulation_posteriors,
        decimal=2,
    )
