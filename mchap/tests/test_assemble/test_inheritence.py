import pytest
import numpy as np

from mchap import mset
from mchap.assemble import inheritence

def test_gamete_probabilities__hom():

    genotypes = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    ], np.int8)
    probabilities = np.array([1])

    gametes_expect = np.array([
        [[0, 0, 0],
         [0, 0, 0]]
    ], np.int8)
    probs_expect = np.array([1])

    gametes_actual, probs_actual = inheritence.gamete_probabilities(
        genotypes,
        probabilities,
    )
    assert mset.equal(gametes_expect, gametes_actual)
    np.testing.assert_array_equal(probs_expect, probs_actual)


def test_gamete_probabilities__het():

    genotypes = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1]]
    ], np.int8)
    probabilities = np.array([1])

    gametes_expect = np.array([
        [[0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 1, 1]],
        [[1, 1, 1],
         [1, 1, 1]]
    ], np.int8)
    probs_expect = np.array([1/6, 4/6, 1/6])

    gametes_actual, probs_actual = inheritence.gamete_probabilities(
        genotypes,
        probabilities,
    )
    assert mset.equal(gametes_expect, gametes_actual)
    np.testing.assert_array_equal(probs_expect, probs_actual)


def test_gamete_probabilities__distribution():
    genotypes = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 1],
         [1, 1, 1]],
    ], np.int8)
    probabilities = np.array([0.6, 0.3, 0.1])

    gametes_expect = np.array([
        [[0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 1, 1]],
        [[1, 1, 1],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 1, 1]],
        [[0, 1, 1],
         [1, 1, 1]],
    ], dtype=np.int8)
    probs_expect = np.array([
        (0.6 * 3/6) + (0.3 * 1/6) + (0.1 * 1/6),
        (0.6 * 3/6) + (0.3 * 4/6) + (0.1 * 2/6),
        (0.6 * 0/6) + (0.3 * 1/6) + (0.1 * 0/6),
        (0.6 * 0/6) + (0.3 * 0/6) + (0.1 * 2/6),
        (0.6 * 0/6) + (0.3 * 0/6) + (0.1 * 1/6),
    ])

    gametes_actual, probs_actual = inheritence.gamete_probabilities(
        genotypes,
        probabilities,
    )

    assert mset.equal(gametes_expect, gametes_actual)
    np.testing.assert_array_equal(probs_expect, probs_actual)


def test_cross_probabilities__hom_x_het():

    maternal_gametes = np.array([
        [[0, 0, 0],
         [0, 0, 0]]
    ], np.int8)
    maternal_probs = np.array([1])
    maternal_probs = np.array([1])

    paternal_gametes = np.array([
        [[0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [1, 1, 1]]
    ], np.int8)
    paternal_probs = np.array([0.5, 0.5])

    genotypes_expect = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]],
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
    ], dtype=np.int8)
    probs_expect = np.array([0.5, 0.5])

    genotypes_actual, probs_actual = inheritence.cross_probabilities(
        maternal_gametes,
        maternal_probs,
        paternal_gametes,
        paternal_probs,
    )
    assert mset.equal(genotypes_expect, genotypes_actual)
    np.testing.assert_array_equal(probs_expect, probs_actual)


