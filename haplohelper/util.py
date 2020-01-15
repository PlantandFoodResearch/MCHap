#!/usr/bin/env python3

import numpy as np
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
