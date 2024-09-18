import pytest
import numpy as np

from mchap.io.vcf import util


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(None, ".", id="None"),
        pytest.param(np.nan, ".", id="nan"),
        pytest.param("", ".", id="empty-string"),
        pytest.param(7, "7", id="integer"),
        pytest.param((1, 2, 3), "1,2,3", id="tuple-integer"),
        pytest.param([1, 2, 3], "1,2,3", id="list-integer"),
        pytest.param(np.array([1, 2, 3]), "1,2,3", id="array-integer"),
        pytest.param([1, None, 3], "1,.,3", id="list-mixed"),
        pytest.param([1.3, 0.7, 1.0], "1.3,0.7,1", id="list-floats"),
        pytest.param(
            np.array([0.0321, np.nan, 1.0, 0.0]),
            "0.032,.,1,0",
            id="array-floats",  # default precision of 3
        ),
    ],
)
def test_vcfstr(obj, expect):
    actual = util.vcfstr(obj)
    assert actual == expect
