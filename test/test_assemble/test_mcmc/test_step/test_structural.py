import numpy as np
import pytest

from haplohelper.assemble.mcmc.step import structural



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

