import numpy as np

from mchap.io import util


def test_qual_of_char():
    chars = np.array([chr(i + 33) for i in np.arange(61)])
    expect = np.arange(61)
    actual = util.qual_of_char(chars)
    np.testing.assert_array_equal(expect, actual)


def test_prob_of_qual():
    qual = np.arange(0, 61, step=10)
    expect = np.array([0.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
    actual = util.prob_of_qual(qual)
    np.testing.assert_array_equal(expect, actual)


def test_qual_of_prob():
    prob = np.array([0.0, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999])
    expect = np.array([0, 10, 20, 30, 40, 40, 40])
    actual = util.qual_of_prob(prob, precision=4)
    np.testing.assert_array_equal(expect, actual)
