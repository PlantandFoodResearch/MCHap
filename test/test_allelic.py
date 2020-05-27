import pytest
import numpy as np

from haplohelper.encoding import allelic

def test_from_strings():
    strings = ['0000', '0101', '1121', '01-1']
    query = allelic.from_strings(strings)

    answer = np.array([
        [0,0,0,0], 
        [0,1,0,1],
        [1,1,2,1],
        [0,1,-1,1]
    ], dtype=np.int8)

    np.testing.assert_array_equal(query, answer)


def test_as_strings():
    array = np.array([
        [0,0,0,0], 
        [0,1,0,1],
        [1,1,2,1],
        [0,1,-1,1]
    ], dtype=np.int8)
    query = allelic.as_strings(array)

    answer = ['0000', '0101', '1121', '01-1']
    np.testing.assert_array_equal(query, answer)


def test_as_probabilistic():
    array = np.array([
        [0,0,0,0], 
        [0,1,0,1],
        [1,1,2,1],
        [0,-1,-1,1]
    ], dtype=np.int8)

    n_alleles = [2,2,3,2]
    p = 0.7

    answer = np.array([
        [[0.7, 0.3, 0.0], [0.7, 0.3, 0.0], [0.7, 0.15, 0.15], [0.7, 0.3, 0.0]],
        [[0.7, 0.3, 0.0], [0.3, 0.7, 0.0], [0.7, 0.15, 0.15], [0.3, 0.7, 0.0]],
        [[0.3, 0.7, 0.0], [0.3, 0.7, 0.0], [0.15, 0.15, 0.7], [0.3, 0.7, 0.0]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, 0.0], [np.nan, np.nan, np.nan], [0.3, 0.7, 0.0]],
    ])

    query = allelic.as_probabilistic(array, n_alleles=n_alleles, p=p)

    np.testing.assert_almost_equal(query, answer)

    # now try without gaps

    answer[3][1] = np.array([0.5, 0.5, 0.0])
    answer[3][2] = np.array([1/3, 1/3, 1/3])

    query = allelic.as_probabilistic(array, n_alleles=n_alleles, p=p, gaps=False)

    np.testing.assert_almost_equal(query, answer)


def test_argsort():
    # should produce same order as sorting strings
    strings = ['0101', '1121', '01-1', '0000']
    array = allelic.from_strings(strings)

    answer = np.argsort(strings)
    query = allelic.argsort(array)
    np.testing.assert_array_equal(answer, query)


def test_sort():
    # should produce same order as sorting strings
    strings = ['0101', '1121', '01-1', '0000']
    array = allelic.from_strings(strings)

    answer = np.sort(strings)
    query = allelic.as_strings(allelic.sort(array))
    np.testing.assert_array_equal(answer, query)