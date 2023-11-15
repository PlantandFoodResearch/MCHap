import numpy as np
import pytest


from mchap.pedigree.validation import duo_valid, trio_valid


@pytest.mark.parametrize(
    "progeny, parent, tau, lambda_, expect",
    [
        ([0, 0, 0, 0], [0, 0, 0, 0], 2, 0.0, True),
        ([0, 0, 0, 0], [0, 0], 2, 0.0, True),
        ([1, 1, 1, 1], [0, 0, 0, 1], 2, 0.0, False),
        ([1, 1, 1, 1], [0, 0, 0, 1], 2, 0.1, True),
        ([0, 0, 0, 0], [1, 2, 3, 4], 2, 0.0, False),
        ([0, 0, 0, 0], [1, 2, 3, 4], 2, 0.1, False),
        ([0, 0, 0, 0], [0, 0, 3, 4], 3, 0.0, False),
        ([0, 0, 0, 0], [0, 0, 0, 4], 3, 0.0, True),
    ],
)
def test_duo_valid(progeny, parent, tau, lambda_, expect):
    progeny = np.array(progeny)
    parent = np.array(parent)
    assert duo_valid(progeny, parent, tau, lambda_) == expect


@pytest.mark.parametrize(
    "progeny, parent_p, parent_q, tau_p, tau_q, lambda_p, lambda_q, expect",
    [
        ([0, 0], [0, 0], [0, 0], 1, 1, 0, 0, True),
        ([0, 1], [0, 0], [1, 1], 1, 1, 0, 0, True),
        ([0, 1], [0, 3], [1, 2], 1, 1, 0, 0, True),
        ([0, 1], [0, 1], [2, 3], 1, 1, 0, 0, False),
        ([0, 3], [0, 1], [0, 2], 1, 1, 0, 0, False),
        ([0, 1, 2, 3], [0, 1], [2, 3], 2, 2, 0, 0, True),
        ([1, 1, 2, 3], [0, 1], [2, 3], 2, 2, 0, 0, False),
        ([1, 1, 2, 3], [0, 1], [2, 3], 2, 2, 0.5, 0, True),
        ([0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 2, 2, 0, 0, True),
        ([0, 0, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], 2, 2, 0, 0, True),
        ([0, 0, 0, 1], [0, 0, 0, 0], [1, 1, 1, 1], 2, 2, 0, 0, False),
        ([0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], 2, 2, 0, 0, False),
        ([0, 1, 1, 1], [0, 0, 0, 0], [1, 1, 1, 1], 2, 2, 0.1, 0.1, False),
        ([1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 2, 2], 2, 2, 0.1, 0.0, True),
        ([1, 1, 1, 1], [0, 0, 0, 1], [1, 1, 2, 2], 2, 2, 0.0, 0.1, False),
        ([1, 1, 1, 3], [0, 0, 1, 2], [1, 1, 2, 3], 2, 2, 0.0, 0.0, False),
        ([1, 1, 1, 3], [0, 0, 1, 2], [1, 1, 2, 3], 2, 2, 0.1, 0.0, True),
        ([0, 0, 0, 3], [0, 0, 0, 1], [0, 0, 0, 2], 2, 2, 0, 0, False),
        ([0, 0, 0, 3], [0, 0, 0, 1], [0, 0, 0, 2], 2, 2, 0.1, 0.1, False),
        ([0, 1, 1, 2, 2, 3], [0, 0, 1, 2, 2, 2], [1, 1, 1, 1, 2, 3], 3, 3, 0, 0, True),
        ([0, 1, 2, 2, 2, 3], [0, 0, 1, 2, 2, 2], [0, 0, 0, 3, 3, 3], 3, 3, 0, 0, False),
    ],
)
def test_trio_valid(
    progeny, parent_p, parent_q, tau_p, tau_q, lambda_p, lambda_q, expect
):
    progeny = np.array(progeny)
    parent_p = np.array(parent_p)
    parent_q = np.array(parent_q)
    assert (
        trio_valid(
            progeny,
            parent_p,
            parent_q,
            tau_p,
            tau_q,
            lambda_p,
            lambda_q,
        )
        == expect
    )
