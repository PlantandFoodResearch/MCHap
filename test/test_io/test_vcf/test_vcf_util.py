import pytest
import numpy as np

from mchap.io.vcf import util

@pytest.mark.parametrize('obj,expect', [
    pytest.param(None, '.', id='None'),
    pytest.param('', '.', id='empty-string'),
    pytest.param(7, '7', id='integer'),
    pytest.param((1, 2, 3), '1,2,3', id='tuple-integer'),
    pytest.param([1, 2, 3], '1,2,3', id='list-integer'),
    pytest.param(np.array([1, 2, 3]), '1,2,3', id='array-integer'),
    pytest.param([1, None, 3], '1,.,3', id='list-mixed'),
])
def test_vcfstr(obj, expect):
    actual = util.vcfstr(obj)
    assert actual == expect
