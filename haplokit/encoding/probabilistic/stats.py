#!/usr/bin/env python3

import numpy as np

from .sequence import is_gap as _is_gap


def identity_prob(*args, ignore_gaps=False):
    """Probability that sequences are identical by state (IBS).
    """
    func = np.nanprod if ignore_gaps else np.prod
    return func(np.sum(np.prod(args, axis=0), axis=-1), axis=-1)


def hamming_exp(*args, ignore_gaps=False):
    """Expected Hamming distance between sequences.
    """
    func = np.nansum if ignore_gaps else np.sum
    return func(1 - np.sum(np.prod(args, axis=0), axis=-1), axis=-1)
