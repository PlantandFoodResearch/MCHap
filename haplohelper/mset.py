#!/usr/bin/env python3

import numpy as np
from collections import Counter as _Counter


def add(array_x, array_y):
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype

    return np.concatenate([array_x, array_y])


def subtract(array_x, array_y):
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype
    element_shape = array_x.shape[1:]
    
    x_map = {element.tostring(): element for element in array_x}
    x_counts = _Counter(element.tostring() for element in array_x)
    y_counts = _Counter(element.tostring() for element in array_y)
    
    counts = x_counts - y_counts
    shape = (sum(counts.values()), *element_shape)
    
    result = np.empty(shape, array_x.dtype)
    
    i = 0
    for k, v in counts.items():
        for _ in range(v):
            result[i] = x_map[k].copy()
            i += 1
    return result


def intercept(array_x, array_y):
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype
    element_shape = array_x.shape[1:]
    
    x_map = {element.tostring(): element for element in array_x}
    x_counts = _Counter(element.tostring() for element in array_x)
    y_counts = _Counter(element.tostring() for element in array_y)
    
    counts = x_counts & y_counts
    shape = (sum(counts.values()), *element_shape)
    
    result = np.empty(shape, array_x.dtype)
    
    i = 0
    for k, v in counts.items():
        for _ in range(v):
            result[i] = x_map[k].copy()
            i += 1
    return result


def union(array_x, array_y):
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype
    element_shape = array_x.shape[1:]
    
    u_map = {element.tostring(): element for element in array_x}
    u_map.update({element.tostring(): element for element in array_y})
    x_counts = _Counter(element.tostring() for element in array_x)
    y_counts = _Counter(element.tostring() for element in array_y)
    
    counts = x_counts | y_counts
    shape = (sum(counts.values()), *element_shape)
    
    result = np.zeros(shape, array_x.dtype)
    
    i = 0
    for k, v in counts.items():
        for _ in range(v):
            result[i] = u_map[k].copy()
            i += 1
    return result


def equal(array_x, array_y):
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype

    counts_x = _Counter(hash(a.tostring()) for a in array_x)
    counts_y = _Counter(hash(a.tostring()) for a in array_y)

    return counts_x == counts_y


def contains(array_x, array_y):
    """Does x contain (is x a superset of) y.
    """
    counts_x = _Counter(hash(a.tostring()) for a in array_x)
    counts_y = _Counter(hash(a.tostring()) for a in array_y)

    return len(counts_y - counts_x) == 0


def within(array_x, array_y):
    """Is x contained within (a subset of) y.
    """
    counts_x = _Counter(hash(a.tostring()) for a in array_x)
    counts_y = _Counter(hash(a.tostring()) for a in array_y)

    return len(counts_x - counts_y) == 0


def unique_idx(array):
    hashes = {hash(a.tostring()) for a in array}
    idx = np.zeros(len(array)).astype(np.bool)
    for i in range(len(idx)):
        hash_ = hash(array[i].tostring())
        if hash_ in hashes:
            hashes -= {hash_}
            idx[i] = True
    return idx


def unique(array):
    return array[unique_idx(array)]


def count(array, categories):
    assert categories.ndim == array.ndim
    assert categories.dtype == array.dtype
    hashes = _Counter(hash(a.tostring()) for a in array)
    counts = np.zeros(len(categories), dtype=np.int)
    for i, cat in enumerate(categories):
        hash_ = hash(cat.tostring())
        if hash_ in hashes:
            counts[i] = hashes[hash_]
        else:
            counts[i] = 0
    return counts


def unique_counts(array, order=None):
    assert order in {'ascending', 'descending', None}
    cats = unique(array)
    counts = count(array, cats)
    if order is None:
        return cats, counts

    idx = np.argsort(counts)
    if order == 'descending':
        idx = np.flip(idx, axis=0)

    return cats[idx], counts[idx]


def repeat(array, counts):
    assert len(array) == len(counts)
    idx = np.repeat(np.arange(len(counts)), counts)
    return array[idx]
