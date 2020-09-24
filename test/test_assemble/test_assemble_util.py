import numpy as np
import math
import pytest

from mchap.assemble import util


def test_add_log_prob():

    for _ in range(10):
        p1 = np.random.rand()
        p2 = np.random.rand()

        log_p1 = np.log(p1)
        log_p2 = np.log(p2)

        query = util.add_log_prob(log_p1, log_p2)
        answer = np.log(p1 + p2)

        assert np.round(query, 10) == np.round(answer, 10)


def test_sum_log_probs():

    for _ in range(10):
        length = np.random.randint(2, 10)
        p = np.random.rand(length)
        answer = np.log(np.sum(p))

        log_p = np.log(p)
        query = util.sum_log_probs(log_p)

        assert np.round(query, 10) == np.round(answer, 10)


def test_log_likelihoods_as_conditionals():

    for _ in range(10):
        # a vector of likelihoods that don't sum to 1
        length = np.random.randint(2, 10)
        lks = np.random.rand(length)

        # the conditional probabilities are calculated by 
        # normalising the vector
        answer = lks / np.sum(lks)

        # now in log space
        llks = np.log(lks)
        query = util.log_likelihoods_as_conditionals(llks)

        np.testing.assert_almost_equal(query, answer)


def test_log_likelihoods_as_conditionals_zeros():

    # a vector of likelihoods that don't sum to 1
    lks = np.random.rand(10)

    # should be able to handel likelihoods of 0
    lks[5:] = 0

    # the conditional probabilities are calculated by 
    # normalising the vector
    answer = lks / np.sum(lks)

    # now in log space
    # ignore warning for log of 0 which produces -inf
    with np.errstate(divide='ignore'):
        llks = np.log(lks)

    query = util.log_likelihoods_as_conditionals(llks)

    np.testing.assert_almost_equal(query, answer)


@pytest.mark.parametrize('x,y,interval,answer', [
    pytest.param(
        [0, 1, 1, 2, 0, 0, 1], 
        [0, 1, 1, 2, 0, 1, 1], 
        None, 
        False, 
        id='0'),
    pytest.param(
        [0, 1, 1, 2, 0, 0, 1], 
        [0, 1, 1, 2, 0, 1, 1], 
        (0, 5), 
        True, 
        id='1'),
    pytest.param(
        [0, 1, 1, 2, 0, 0, 1], 
        [0, 1, 1, 2, 0, 1, 1], 
        (5, 7), 
        False, 
        id='2'),
    pytest.param(
        [0, 1, 1, 2, 0, 0, 1], 
        [0, 1, 1, 2, 0, 1, 1], 
        (5, 5), # zero width interval
        True, 
        id='3'),
    pytest.param(
        [], 
        [], 
        None, 
        True, 
        id='4'),
])
def test_array_equal(x, y, interval, answer):

    x = np.array(x, dtype=np.int)
    y = np.array(y, dtype=np.int)

    query = util.array_equal(x, y, interval=interval)

    assert query is answer


@pytest.mark.parametrize('genotype,interval,answer', [
    pytest.param(
        [[0, 1, 0], [0, 1, 0]],
        None,
        [2, 0],
        id='2x-hom'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        None,
        [1, 1],
        id='2x-het'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        (0, 2),
        [2, 0],
        id='2x-het-hom-interval'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        (1, 3),
        [1, 1],
        id='2x-het-het-interval'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1]],
        (2, 2),
        [2, 0],
        id='2x-het-zero-width-interval'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1], [0, 1, 0]],
        None,
        [2, 1, 0],
        id='3x-2:1'),
    pytest.param(
        [[0, 1, 0], [0, 1, 1], [0, 1, 1]],
        None,
        [1, 2, 0],
        id='3x-1:2'),
    pytest.param(
        [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
        None,
        [1, 2, 0, 1],
        id='4x-1:2:1'),
    pytest.param(
        [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]],
        (0, 3),
        [2, 2, 0, 0],
        id='4x-2:2-interval'),
])
def test_get_dosage(genotype, interval, answer):

    genotype = np.array(genotype, dtype=np.int8)
    answer = np.array(answer)

    ploidy = len(genotype)
    dosage = np.ones(ploidy, dtype=np.int)

    util.get_dosage(dosage, genotype, interval=interval)

    np.testing.assert_almost_equal(dosage, answer)


def set_dosage():

    # initial dosage = [1, 2, 0, 1]
    genotype = np.array(
        [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 1, 1], [0, 1, 0, 0]], 
        dtype=np.int8
    )

    # target dosage
    dosage = np.array([3, 1, 0, 0], dtype=np.int)

    util.set_dosage(genotype, dosage)

    # note first haplotypes in same order
    answer = np.array(
        [[0, 1, 0, 1], [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 1]], 
        dtype=np.int8
    )

    np.testing.assert_array_equal(genotype, answer)


def test_factorial_20():

    for i in range(0, 21):
        assert util.factorial_20(i) == math.factorial(i)


@pytest.mark.parametrize('dosage,answer', [
    pytest.param([2, 0], 1),
    pytest.param([1, 1], 2),
    pytest.param([0, 2], 1),
    pytest.param([1, 1, 1], 6),
    pytest.param([2, 1, 0], 3),
    pytest.param([3, 0, 0], 1),
    pytest.param([4, 0, 0, 0], 1),
    pytest.param([2, 2, 0, 0], 6),
    pytest.param([1, 1, 1, 1], 24),
])
def test_count_equivalent_permutations(dosage, answer):
    dosage = np.array(dosage, dtype=np.int)
    query = util.count_equivalent_permutations(dosage)
    assert query == answer
