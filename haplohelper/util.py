#!/usr/bin/env python3

import numpy as np

from scipy import stats as _stats
from itertools import islice as _islice
from itertools import zip_longest as _zip_longest

import biovector


def prob_of_qual(qual):
    return 1 - (10 ** (qual / -10))


def qual_of_prob(prob):
    return int(-10 * np.log10((1 - prob)))


def flatten(item, container):
    if isinstance(item, container):
        for element in item:
            yield from flatten(element, container)
    else:
        yield item


def merge(*args):
    for tup in _zip_longest(*args):
        for val in tup:
            if val is not None:
                yield val


def middle_out(sequence):
    gen = (i for i in sequence)
    first_half = list(_islice(gen, len(sequence) // 2))
    second_half = list(gen)
    return merge(second_half, reversed(first_half))


def point_beta_probabilities(n_base, a=1, b=1):
    """Return probabilies for selecting a recombination point
    following a beta distribution

    Parameters
    ----------
    n_base : int
        Number of base positions in this genotype.
    a : float
        Alpha parameter for beta distribution.
    b : float
        Beta parameter for beta distribution.

    Returns
    -------
    probs : array_like, int, shape (n_base - 1)
        Probabilities for recombination point.
    
    """
    dist = _stats.beta(a, b)
    points = np.arange(1, n_base + 1) / (n_base)
    probs = dist.cdf(points)
    probs[1:] = probs[1:] - probs[:-1]
    return probs


def tile_to_shape(x, shape):
    ndims = np.ndim(x)
    diff = len(shape) - ndims
    assert np.shape(x) == shape[diff:]
    template = shape[:diff] + tuple(1 for _ in range(ndims))
    return np.tile(x, template)
