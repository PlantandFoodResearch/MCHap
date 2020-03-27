import pytest
import numpy as np

from haplohelper.encoding import allelic, probabilistic


def test_is_gap():

    array = np.array([
        [[0.7, 0.15, 0.15], [1/3, 1/3, 1/3]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],
        [[0.7, 0.3, 0.0], [0.4, 0.4, 0.3]],  # 2nd dist is invalid
        [[np.nan, np.nan, np.nan], [np.nan, 0.7, 0.3]]  # 2nd dist is invalid
    ])

    answer = np.array([
        [False, False],
        [False, True],
        [False, False],
        [True, False]], 
        dtype=np.bool)

    query = probabilistic.is_gap(array)

    np.testing.assert_array_equal(query, answer)


def test_is_dist():

    array = np.array([
        [[0.7, 0.15, 0.15], [1/3, 1/3, 1/3]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],
        [[0.7, 0.3, 0.0], [0.4, 0.4, 0.3]],  # 2nd dist is invalid
        [[np.nan, np.nan, np.nan], [np.nan, 0.7, 0.3]]  # 2nd dist is invalid
    ])

    answer = np.array([
        [True, True],
        [True, False],
        [True, False],
        [False, False]], 
        dtype=np.bool)

    query = probabilistic.is_dist(array)

    np.testing.assert_array_equal(query, answer)



def test_is_valid():

    array = np.array([
        [[0.7, 0.15, 0.15], [1/3, 1/3, 1/3]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],
        [[0.7, 0.3, 0.0], [0.4, 0.4, 0.3]],  # 2nd dist is invalid
        [[np.nan, np.nan, np.nan], [np.nan, 0.7, 0.3]]  # 2nd dist is invalid
    ])

    answer = np.array([
        [True, True],
        [True, True],
        [True, False],
        [True, False]], 
        dtype=np.bool)

    query = probabilistic.is_valid(array)

    np.testing.assert_array_equal(query, answer)


def test_depth():

    array = np.array([
        [[0.7, 0.15, 0.15], [1/3, 1/3, 1/3]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],  
        [[np.nan, np.nan, np.nan], [0.15, 0.7, 0.15]]  
    ])

    answer = np.array([3, 2], dtype=np.int)

    query = probabilistic.depth(array)

    np.testing.assert_array_equal(query, answer)


def test_allele_depth():

    array = np.array([
        [[0.7, 0.15, 0.15], [1/3, 1/3, 1/3]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan]],  
        [[np.nan, np.nan, np.nan], [0.15, 0.7, 0.15]]  
    ])

    answer = np.array([
        [2.1, 0.75, 0.15],
        [0.15+1/3,  0.7+1/3, 0.15+1/3]
    ], dtype=np.float)

    query = probabilistic.allele_depth(array)

    np.testing.assert_array_almost_equal(query, answer)




def test_call_alleles():
    array = np.array([
        [[0.7, 0.3, 0.0], [0.7, 0.3, 0.0], [0.7, 0.15, 0.15], [0.7, 0.3, 0.0]],
        [[0.99, 0.01, 0.0], [0.05, 0.95, 0.0], [0.7, 0.15, 0.15], [0.3, 0.7, 0.0]],
        [[0.05, 0.95, 0.0], [0.06, 0.94, 0.0], [0.01, 0.01, 0.98], [0.1, 0.99, 0.0]],
        [[0.7, 0.3, 0.0], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [0.1, 0.9, 0.0]],
    ])

    answer = np.array([
        [-1, -1, -1, -1],
        [ 0,  1, -1, -1],
        [ 1, -1,  2,  1],
        [-1, -1, -1, -1]], 
        dtype=np.int8
    )

    query = probabilistic.call_alleles(array, p=0.95)

    np.testing.assert_array_equal(query, answer)


def test_sample_alleles():

    array = np.array([
        [[0.7, 0.3, 0.0], [0.5, 0.5, 0.0]],
        [[0.9, 0.1, 0.0], [0.4, 0.3, 0.3]]
    ])

    acumulate = np.zeros(array.shape, dtype=np.int)
    for _ in range(10000):
        acumulate += allelic.as_probabilistic(probabilistic.sample_alleles(array), 3, dtype=np.int)

    # should be no samples from zero probability alleles
    assert acumulate[0][0][-1] == 0
    
    # should reproduce origional array
    query = np.round(acumulate/10000, 1)
    np.testing.assert_array_equal(query, array)

@pytest.mark.parametrize('idx_0,idx_1,answer,ignore_gaps', [
    pytest.param(0, 0, 0.314432, False, id='0'),
    pytest.param(0, 1, 0.147968, False, id='1'),
    pytest.param(0, 2, np.nan, False, id='nan'),
    pytest.param(0, 2, 0.4624, True, id='ignore_nan'),
    pytest.param([0,1,2], [0,1,2], np.array([0.314432, 0.314432, np.nan]), False, id='array'),
])
def test_identity_prob(idx_0, idx_1, answer, ignore_gaps):

    array = np.array([
        [[0.8, 0.2],
         [0.8, 0.2],
         [0.8, 0.2]],
        [[0.8, 0.2],
         [0.8, 0.2],
         [0.2, 0.8]],
        [[0.8, 0.2],
         [0.8, 0.2],
         [np.nan, np.nan]]
    ])

    query = probabilistic.identity_prob(
        array[idx_0], 
        array[idx_1], 
        ignore_gaps=ignore_gaps
    )

    if np.shape(answer) is ():
        # scalar answer
        if np.isnan(answer):  
            # scalar nan answer
            assert np.isnan(query)
        else: 
            # scalar float answer
            assert np.round(query, 10) == answer
    else: 
        # array answer
        np.testing.assert_array_almost_equal(query, answer)

    
@pytest.mark.parametrize('idx_0,idx_1,answer,ignore_gaps', [
    pytest.param(0, 0, 0.96, False, id='0'),
    pytest.param(0, 1, 1.32, False, id='1'),
    pytest.param(0, 2, np.nan, False, id='nan'),
    pytest.param(0, 2, 0.64, True, id='ignore_nan'),
    pytest.param([0,1,2], [0,1,2], np.array([0.96, 0.96, np.nan]), False, id='array'),
])
def test_hamming_exp(idx_0, idx_1, answer, ignore_gaps):

    array = np.array([
        [[0.8, 0.2],
         [0.8, 0.2],
         [0.8, 0.2]],
        [[0.8, 0.2],
         [0.8, 0.2],
         [0.2, 0.8]],
        [[0.8, 0.2],
         [0.8, 0.2],
         [np.nan, np.nan]]
    ])

    query = probabilistic.hamming_exp(
        array[idx_0], 
        array[idx_1], 
        ignore_gaps=ignore_gaps
    )

    if np.shape(answer) is ():
        # scalar answer
        if np.isnan(answer):  
            # scalar nan answer
            assert np.isnan(query)
        else: 
            # scalar float answer
            assert np.round(query, 10) == answer
    else: 
        # array answer
        np.testing.assert_array_almost_equal(query, answer)

