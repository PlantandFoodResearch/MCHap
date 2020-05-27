import numpy as np
import pytest

from haplokit.assemble.mcmc.step import structural


@pytest.mark.parametrize('breaks,n', [
    pytest.param(0, 1),
    pytest.param(2, 5),
    pytest.param(0, 11),
    pytest.param(1, 11),
    pytest.param(3, 11),
    pytest.param(10, 11),
        pytest.param(20, 100),
])
def test_random_breaks(breaks, n):

    # intervals are of random length but should be within several constraints
    # so test that they 
    intervals = structural.random_breaks(breaks, n)

    # number of intervals produced is 1 + the break points between intervals
    assert len(intervals) == breaks + 1

    # all intervals must be of length 1 or greater
    lengths = intervals[:,1] - intervals[:,0]
    assert np.all(lengths > 0)

    # all positions must be contained in one and only one interval
    # i.e. the end position of one interval must also be the start of the next

    # check first interval starts at 0
    assert intervals[0, 0] == 0

    # check last interval ends at end point
    assert intervals[-1, 1] == n

    # check all intervalls are adjacent 
    stops = intervals[1:,0]  # all but first interval
    starts = intervals[:-1,1] # all but last interval
    np.testing.assert_array_equal(stops, starts)


@pytest.mark.parametrize('genotype,haplotype_indices,interval,answer', [
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        [0, 1, 2, 3],  # keep same haplotypes
        None,
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        id='4x-no-change'),
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        [2, 1, 0, 3],  # switch hap 0 with hap 2 (this is a meaningless change)
        None,
        [[0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1], [0, 1, 1, 2, 1, 0]],
        id='4x-switch'),
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        [0, 1, 0, 3], # overwrite hap 2 with hap 0
        None,         # full haplotype
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 0, 1, 1, 1], [0, 1, 1, 2, 1, 0]],
        id='4x-overwrite'),
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        [3, 1, 2, 3],  # overwrite hap 0 with hap 3
        (3, 5),        # within this interval 
        [[0, 1, 0, 2, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        id='4x-partial-overwrite'),
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        [3, 1, 2, 0],  # switch hap 0 with hap 3
        (3, 5),        # within this interval 
        [[0, 1, 0, 2, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]],
        id='4x-recombine'),
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 2, 1, 0]],
        [2, 3, 3, 1],  # madness
        (3, 6),        # within this interval 
        [[0, 1, 0, 1, 1, 0], [0, 1, 1, 2, 1, 0], [0, 1, 1, 2, 1, 0], [0, 1, 1, 1, 1, 1]],
        id='4x-multi'),
])
def test_structural_change(genotype, haplotype_indices, interval, answer):

    genotype = np.array(genotype, dtype=np.int8)
    haplotype_indices = np.array(haplotype_indices, dtype=np.int8)
    answer = np.array(answer, dtype=np.int)

    structural.structural_change(genotype, haplotype_indices, interval=interval)

    np.testing.assert_array_equal(genotype, answer)



@pytest.mark.parametrize('genotype,interval,answer', [
    pytest.param(
        [[0, 1, 0], [0, 1, 0]],
        None,
        [[0, 0], [0, 0]],
        id='2x-hom'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        None,
        [[0, 0], [1, 0]],
        id='2x-het'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        (0, 2),
        [[0, 0], [0, 1]],
        id='2x-het-hom-interval'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        (1, 3),
        [[0, 0], [1, 0]],
        id='2x-het-het-interval'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        (2, 2),
        [[0, 0], [0, 1]],
        id='2x-het-zero-width-interval'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
        None,
        [[0, 0], [1, 0], [0, 0]],
        id='3x-2:1'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1], [0, 1, 1]],
        None,
        [[0, 0], [1, 0], [1, 0]],
        id='3x-1:2'),
    pytest.param(
        [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
        None,
        [[0, 0], [1, 0], [1, 0], [3, 0]],
        id='4x-1:2:1'),
    pytest.param(
        [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
        (0, 3),
        [[0, 0], [1, 0], [1, 0], [0, 3]],
        id='4x-2:2-interval'),
    pytest.param(
        [[0, 1, 0, 1, 1, 1], [0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0], [0, 1, 1, 1, 1, 0]],
        (2, 5),
        [[0, 0], [1, 0], [1, 2], [1, 2]],
        id='4x-1:3-interval'),
])
def test_haplotype_segment_labels(genotype, interval, answer):

    genotype = np.array(genotype, dtype=np.int8)
    answer = np.array(answer, dtype=np.int)

    query = structural.haplotype_segment_labels(genotype, interval=interval)

    np.testing.assert_array_equal(query, answer)


@pytest.mark.parametrize('labels,answer', [
    pytest.param(
        [[0, 0], [1, 0]],
        np.empty((0, 2), dtype=np.int8),  # no options
        id='2x-0'),
    pytest.param(
        [[0, 0], [1, 1]],
        [[1, 0]],  # 1 option
        id='2x-1'),
    pytest.param(
        [[0, 0], [0, 1], [0, 1], [0, 0]],
        np.empty((0, 4), dtype=np.int8),  # no options
        id='4x-0'),
    pytest.param(
        [[0, 0], [0, 1], [2, 1], [0, 0]],
        [[2, 1, 0, 3]],  # 1 option
        id='4x-1'),
    pytest.param(
        [[0, 0], [0, 1], [2, 1], [3, 0]],
        [[2, 1, 0, 3],
         [0, 3, 2, 1],
         [0, 1, 3, 2]],  # 3 options
        id='4x-3'),
    pytest.param(
        [[0, 0], [1, 1], [2, 2], [3, 3]],
        [[1, 0, 2, 3],
         [2, 1, 0, 3],
         [3, 1, 2, 0],
         [0, 2, 1, 3],
         [0, 3, 2, 1],
         [0, 1, 3, 2]],  # all the options
        id='4x-6'),
])
def test_recombination_step_options(labels, answer):

    labels = np.array(labels)
    answer = np.array(answer)

    n_options = structural.recombination_step_n_options(labels)
    query = structural.recombination_step_options(labels)

    assert len(query) == n_options
    np.testing.assert_array_equal(query, answer)



@pytest.mark.parametrize('labels,allow_deletions,answer', [
    pytest.param(
        [[0, 0], [0, 0]],
        False,
        np.empty((0, 2), dtype=np.int8),  # no options
        id='2x-hom'),
    pytest.param(
        [[0, 0], [0, 0]],
        True,
        np.empty((0, 2), dtype=np.int8),  # no options
        id='2x-hom-del'),
    pytest.param(
        [[0, 0], [1, 0]],
        False,
        np.empty((0, 2), dtype=np.int8),  # no options
        id='2x-0'),
    pytest.param(
        [[0, 0], [1, 0]],
        True,
        [[1, 1],
         [0, 0]],  # no options
        id='2x-del-2'),
    pytest.param(
        [[0, 0], [0, 0], [0, 0], [3, 0]],
        False,
        [[3, 1, 2, 3]],
        id='4x-1'),
    pytest.param(
        [[0, 0], [0, 0], [0, 0], [3, 0]],
        True,
        [[3, 1, 2, 3],
         [0, 1, 2, 0]],
        id='4x-del-2'),
    pytest.param(
        [[0, 0], [0, 1], [2, 0], [2, 0]],
        False,
        [[2, 1, 2, 3],
         [0, 2, 2, 3],
         [0, 1, 0, 3]],
        id='4x-3'),
    pytest.param(
        [[0, 0], [0, 1], [2, 0], [2, 0]],
        True,
        [[2, 1, 2, 3],
         [0, 2, 2, 3],
         [0, 1, 0, 3]],
        id='4x-del-3'),
])
def test_dosage_step_options(labels, allow_deletions, answer):

    labels = np.array(labels)
    answer = np.array(answer)

    n_options = structural.dosage_step_n_options(labels, allow_deletions)
    query = structural.dosage_step_options(labels, allow_deletions)

    assert len(query) == n_options
    np.testing.assert_array_equal(query, answer)
