import numpy as np
import pytest

from mchap.encoding import integer


def test_from_strings():
    strings = ["0000", "0101", "1121", "01-1"]
    query = integer.from_strings(strings)

    answer = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 2, 1], [0, 1, -1, 1]], dtype=np.int8
    )

    np.testing.assert_array_equal(query, answer)


def test_as_strings():
    array = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 2, 1], [0, 1, -1, 1]], dtype=np.int8
    )
    query = integer.as_strings(array)

    answer = ["0000", "0101", "1121", "01-1"]
    np.testing.assert_array_equal(query, answer)


def test_as_probabilistic():
    array = np.array(
        [[0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 2, 1], [0, -1, -1, 1]], dtype=np.int8
    )

    n_alleles = [2, 2, 3, 2]
    p = 0.7

    answer = np.array(
        [
            [[0.7, 0.1, 0.0], [0.7, 0.1, 0.0], [0.7, 0.1, 0.1], [0.7, 0.1, 0.0]],
            [[0.7, 0.1, 0.0], [0.1, 0.7, 0.0], [0.7, 0.1, 0.1], [0.1, 0.7, 0.0]],
            [[0.1, 0.7, 0.0], [0.1, 0.7, 0.0], [0.1, 0.1, 0.7], [0.1, 0.7, 0.0]],
            [
                [0.7, 0.1, 0.0],
                [np.nan, np.nan, 0.0],
                [np.nan, np.nan, np.nan],
                [0.1, 0.7, 0.0],
            ],
        ]
    )

    query = integer.as_probabilistic(array, n_alleles=n_alleles, p=p)

    np.testing.assert_almost_equal(query, answer)


def test_argsort():
    # should produce same order as sorting strings
    strings = ["0101", "1121", "01-1", "0000"]
    array = integer.from_strings(strings)

    answer = np.argsort(strings)
    query = integer.argsort(array)
    np.testing.assert_array_equal(answer, query)


def test_sort():
    # should produce same order as sorting strings
    strings = ["0101", "1121", "01-1", "0000"]
    array = integer.from_strings(strings)

    answer = np.sort(strings)
    query = integer.as_strings(integer.sort(array))
    np.testing.assert_array_equal(answer, query)


def test_minimum_error_correction():
    reads = np.array(
        [
            [0, 0, 0, 0, 0],  # 0
            [0, 0, 0, -1, -1],  # 0
            [0, 0, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1],  # 0
        ]
    )

    genotype = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )

    query = integer.minimum_error_correction(reads, genotype)
    answer = np.array([0, 0, 2, 0])
    np.testing.assert_array_equal(answer, query)


def read_assignment():
    reads = np.array(
        [
            [0, 0, 0, 0, 0],  # 0
            [0, 0, 0, -1, -1],  # 0
            [0, 0, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1],  # 0
        ]
    )

    genotype = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    query = integer.read_assignment(reads, genotype)
    answer = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_array_equal(answer, query)


@pytest.mark.parametrize(
    "reads,genotype,ks,expect",
    [
        (
            [
                [0, 0],
                [1, 1],
                [1, 1],
            ],
            [
                [0, 0],
                [1, 1],
            ],
            [1, 2, 3],
            [1, 1, np.nan],
        ),
        (
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1],
            ],
            [
                [1, 1, 1],
                [1, 1, 1],
            ],
            [1, 2, 3],
            [1, 1, 1],
        ),
        (
            [
                [1, 1, 1, 1, -1],  # matching
                [1, 1, 1, 1, -1],
                [1, 1, 1, 1, -1],
                [1, 2, 1, -1, -1],  # non-matching
                [1, 2, 1, -1, -1],
                [1, 2, 1, -1, -1],
            ],
            [
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 2, 2, 1, 1],
                [1, 2, 2, 1, 1],
            ],
            [1, 2, 3, 4, 5, 6],
            [1, 0.5, 0.5, 1, np.nan, np.nan],
        ),
    ],
)
def test_min_kmer_coverage(reads, genotype, ks, expect):
    reads = np.array(reads, int)
    genotype = np.array(genotype, int)
    expect = np.array(expect, float)
    actual = integer.min_kmer_coverage(reads, genotype, ks=ks)
    print(actual)
    np.testing.assert_array_equal(expect, actual)
