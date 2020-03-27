import pytest
import numpy as np
from collections import Counter

from haplohelper import mset
from haplohelper.encoding.allelic import from_strings, as_strings


def test_add():
    query_x = ['000', '000', '111', '001']
    query_y = ['000', '111', '111', '010']

    answer = Counter(query_x) + Counter(query_y)

    array_x = from_strings(query_x)
    array_y = from_strings(query_y)

    result = mset.add(array_x, array_y)

    assert Counter(as_strings(result)) == answer


def test_subtract():
    query_x = ['000', '000', '111', '001']
    query_y = ['000', '111', '111', '010']

    answer = Counter(query_x) - Counter(query_y)

    array_x = from_strings(query_x)
    array_y = from_strings(query_y)

    result = mset.subtract(array_x, array_y)

    assert Counter(as_strings(result)) == answer


def test_union():
    query_x = ['000', '000', '111', '001']
    query_y = ['000', '111', '111', '010']

    answer = Counter(query_x) | Counter(query_y)

    array_x = from_strings(query_x)
    array_y = from_strings(query_y)

    result = mset.union(array_x, array_y)

    assert Counter(as_strings(result)) == answer


def test_intercept():
    query_x = ['000', '000', '111', '001']
    query_y = ['000', '111', '111', '010']

    answer = Counter(query_x) & Counter(query_y)

    array_x = from_strings(query_x)
    array_y = from_strings(query_y)

    result = mset.intercept(array_x, array_y)

    assert Counter(as_strings(result)) == answer


@pytest.mark.parametrize('query_x,query_y,answer', [
    pytest.param(['000', '000', '111', '001'], ['000', '000', '111', '001'], True, id='0'),
    pytest.param(['000', '000', '111', '001'], ['000', '111', '001', '000'], True, id='1'),
    pytest.param(['000', '000', '111', '001'], ['000', '111', '111', '001'], False, id='2'),
    pytest.param(['000', '000', '111', '001'], ['000', '000', '111', '011'], False, id='2'),
])
def test_equal(query_x, query_y, answer):
    array_x = from_strings(query_x)
    array_y = from_strings(query_y)
    assert mset.equal(array_x, array_y) is answer


@pytest.mark.parametrize('query_x,query_y,answer', [
    pytest.param(['000', '000', '111', '001'], ['000', '000', '111', '001'], True, id='0'),
    pytest.param(['000', '001'], ['000', '000', '111', '001'], True, id='1'),
    pytest.param(['000', '000', '111', '001'], ['000', '001'], False, id='2'),
    pytest.param(['000', '011'], ['000', '000', '111', '001'], False, id='3'),
])
def test_within(query_x, query_y, answer):
    array_x = from_strings(query_x)
    array_y = from_strings(query_y)
    assert mset.within(array_x, array_y) is answer


@pytest.mark.parametrize('query_x,query_y,answer', [
    pytest.param(['000', '000', '111', '001'], ['000', '000', '111', '001'], True, id='0'),
    pytest.param(['000', '001'], ['000', '000', '111', '001'], False, id='1'),
    pytest.param(['000', '000', '111', '001'], ['000', '001'], True, id='2'),
    pytest.param(['000', '000', '111', '001'], ['000', '011'], False, id='3'),
])
def test_contains(query_x, query_y, answer):
    array_x = from_strings(query_x)
    array_y = from_strings(query_y)
    assert mset.contains(array_x, array_y) is answer


def test_unique():
    query = from_strings(['000', '000', '001'])
    answer = from_strings(['000', '001'])
    result = mset.unique(query)
    assert mset.equal(result, answer)


def test_unique_counts():
    query = from_strings(
        ['111', '000', '100', '000', '000', '100']
    )

    answer_cats = from_strings(['000', '100', '111'])
    answer_counts = np.array([3, 2, 1])

    result_cats, result_counts = mset.unique_counts(query, order='descending')
    np.testing.assert_array_equal(result_cats, answer_cats)
    np.testing.assert_array_equal(result_counts, answer_counts)
