#!/usr/bin/env python3

import numpy as np


# NGS error rate per base estimated by Pfeiffer et al 2018
# "Systematic evaluation of error rates and causes in short samples in next-generation sequencing"
PFEIFFER_ERROR = .0024


def qual_of_char(char):
    if isinstance(char, str):
        qual = ord(char) - 33
        return qual
    elif isinstance(char, np.ndarray):
        if char.dtype == np.dtype('<U1'):
            qual = char.copy()
            qual.dtype = np.int32
            qual -= 33
            return qual
        else:
            raise ValueError('Array must have dtype "<U1"')
    else:
        raise ValueError('Input must be character or array of characters')


def prob_of_qual(qual):
    return 1 - (10 ** (qual / -10))


def qual_of_prob(prob, precision=6):
    # cant have a prob of 1 converted to a qual
    # instead convert based on the expected decimal precision of the prob
    # a precision of 6 produces a max qual of 60
    maximum = 1 - 0.1 ** precision

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
    return np.round((-10 * np.log10((1 - prob)))).astype(np.int)