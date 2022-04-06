#!/usr/bin/env python3

import numpy as np

__all__ = [
    "qual_of_char",
    "prob_of_qual",
    "qual_of_prob",
]


def qual_of_char(char):
    """Convert unicode characters of a qual string into an integer value.

    Paramters
    ---------
    char : array_like
        A single char or array of chars.

    Returns
    -------
    qual : array_like
        A single int or array of integers.
    """
    if isinstance(char, str):
        qual = ord(char) - 33
        return qual
    elif isinstance(char, np.ndarray):
        if char.dtype == np.dtype("<U1"):
            qual = char.copy()
            qual.dtype = np.int32
            qual -= 33
            return qual
        else:
            raise ValueError('Array must have dtype "<U1"')
    else:
        raise ValueError("Input must be character or array of characters")


def prob_of_qual(qual):
    """Convert phred-scaled quality integer into a probability of the call being correct.

    Paramters
    ---------
    qual : array_like
        A single int or array of integers.

    Returns
    -------
    prob : array_like
        A single float or array of floats.
    """
    return 1 - (10 ** (qual / -10))


def qual_of_prob(prob, precision=6):
    """Convert a probability of a call being correct into a phred-scaled quality integer.

    Paramters
    ---------
    prob : array_like
        A single float or array of floats.
    precision : int
        Max precision to treat the probability

    Returns
    -------
    qual : array_like
        A single int or array of integers.
    """
    # cant have a prob of 1 converted to a qual
    # instead convert based on the expected decimal precision of the prob
    # a precision of 6 produces a max qual of 60
    maximum = 1 - 0.1**precision

    if np.shape(prob) == ():
        # prob is scalar
        if prob > maximum:
            prob = maximum
        else:
            pass

    else:
        # prob is array-like
        prob = np.array([maximum if p > maximum else p for p in prob])

    prob = np.floor(prob * 10**precision) / 10**precision
    return np.round((-10 * np.log10((1 - prob)))).astype(int)
