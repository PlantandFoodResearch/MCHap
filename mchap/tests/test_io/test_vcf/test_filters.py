import numpy as np
import pytest

from mchap.io.vcf import filters

@pytest.mark.parametrize('obj,expect', [
    pytest.param(filters.FilterCall('pp95', failed=False, applied=True), 'PASS', id='pass'),
    pytest.param(filters.FilterCall('pp95', failed=True, applied=True), 'pp95', id='fail'),
    pytest.param(filters.FilterCall('pp95', failed=False, applied=False), '.', id='not-applied'),
])
def test_FilterCall__str(obj, expect):
    actual = str(obj)
    assert actual == expect


def test_FilterCallSet__str():
    calls = filters.FilterCallSet((
        filters.FilterCall('pp95', failed=True, applied=True),
        filters.FilterCall('3m95', failed=True, applied=True),
        filters.FilterCall('dp5', failed=False, applied=True),
        filters.FilterCall('nan', failed=False, applied=False),
    ))
    expect = 'pp95,3m95'
    actual = str(calls)
    assert expect == actual


def test_SamplePassFilter():
    obj = filters.SamplePassFilter()
    expect = '##FILTER=<ID=PASS,Description="All filters passed">'
    assert str(obj) == expect
    assert obj.id == 'PASS'


@pytest.mark.parametrize('obj,expect', [
    pytest.param(filters.SampleKmerFilter(k=3, threshold=0.75), 'PASS', id='pass'),
    pytest.param(filters.SampleKmerFilter(k=3, threshold=0.99), '3m99', id='fail'),
    pytest.param(filters.SampleKmerFilter(k=5, threshold=0.95), '.', id='not-applied'),
])
def test_SampleKmerFilter(obj, expect):
    genotype = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, 1],
        [1, 1, 1, 1]
    ])
    read_variants = np.array([
        [-1, -1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 1, 0, -1],
        [1, -1, 1, 1],
        [0, 1, 1, -1],  # 011- is not in haplotypes
    ])
    # apply filter and get FT code
    actual = str(obj(read_variants, genotype))
    assert actual == expect


@pytest.mark.parametrize('obj,depth,expect', [
    pytest.param(
        filters.SampleDepthFilter(threshold=5), 
        np.array([5,6,6,5]),
        'PASS', 
        id='pass'
    ),
    pytest.param(
        filters.SampleDepthFilter(threshold=6), 
        np.array([5,6,6,5]),
        'dp6', 
        id='fail'
    ),
    pytest.param(
        filters.SampleDepthFilter(threshold=6), 
        np.array([]),
        '.', 
        id='not-applied'
    ),
])
def test_SampleDepthFilter(obj, depth, expect):
    actual = str(obj(depth))
    assert actual == expect


@pytest.mark.parametrize('obj,expect', [
    pytest.param(filters.SampleReadCountFilter(threshold=5), 'PASS', id='pass'),
    pytest.param(filters.SampleReadCountFilter(threshold=6), 'rc6', id='fail'),
])
def test_SampleReadCountFilter(obj, expect):
    count = 5
    actual = str(obj(count))
    assert actual == expect


@pytest.mark.parametrize('obj,expect', [
    pytest.param(filters.SamplePhenotypeProbabilityFilter(threshold=0.95), 'PASS', id='pass'),
    pytest.param(filters.SamplePhenotypeProbabilityFilter(threshold=0.99), 'pp99', id='fail'),
])
def test_SamplePhenotypeProbabilityFilter(obj, expect):
    prob = 0.9773
    actual = str(obj(prob))
    assert actual == expect