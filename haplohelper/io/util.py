#!/usr/bin/env python3

import numpy as np


# NGS error rate per base estimated by Pfeiffer et al 2018
# "Systematic evaluation of error rates and causes in short samples in next-generation sequencing"
PFEIFFER_ERROR = .0024


def qual_of_char(char):
    return ord(char) - 33


def prob_of_qual(qual):
    return 1 - (10 ** (qual / -10))


def qual_of_prob(prob):
    return int(-10 * np.log10((1 - prob)))
