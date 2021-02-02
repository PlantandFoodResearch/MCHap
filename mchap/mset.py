#!/usr/bin/env python3

import numpy as np
from collections import Counter as _Counter


def add(array_x, array_y):
    """Multi-set addition of a pair of n-dimentional arrays.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    array : ndarray, int
        The disjoint union of inputs outer dimention.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype

    return np.concatenate([array_x, array_y])


def subtract(array_x, array_y):
    """Multi-set subtraction of a pair of n-dimentional arrays.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    array : ndarray, int
        The complement of elements of array_y in array_x.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype
    element_shape = array_x.shape[1:]
    
    x_map = {element.tobytes(): element for element in array_x}
    x_counts = _Counter(element.tobytes() for element in array_x)
    y_counts = _Counter(element.tobytes() for element in array_y)
    
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
    """Multi-set intercept of a pair of n-dimentional arrays.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    array : ndarray, int
        The intercept of elements of the input arrays.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype
    element_shape = array_x.shape[1:]
    
    x_map = {element.tobytes(): element for element in array_x}
    x_counts = _Counter(element.tobytes() for element in array_x)
    y_counts = _Counter(element.tobytes() for element in array_y)
    
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
    """Multi-set union of a pair of n-dimentional arrays.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    array : ndarray, int
        The union of elements of the input arrays.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype
    element_shape = array_x.shape[1:]
    
    u_map = {element.tobytes(): element for element in array_x}
    u_map.update({element.tobytes(): element for element in array_y})
    x_counts = _Counter(element.tobytes() for element in array_x)
    y_counts = _Counter(element.tobytes() for element in array_y)
    
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
    """Test multi-set equality of a pair of n-dimentional arrays.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    equality : bool
        The multi-set equality of the input arrays.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    assert array_x.ndim == array_y.ndim
    assert array_x.dtype == array_y.dtype

    counts_x = _Counter(a.tobytes() for a in array_x)
    counts_y = _Counter(a.tobytes() for a in array_y)

    return counts_x == counts_y


def contains(array_x, array_y):
    """Check if the elements of array_x are a super-set 
    of the elements of array_y.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    superset : bool
        `True` if elements of array_x are a super-set
        of the elements of array_y.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    counts_x = _Counter(a.tobytes() for a in array_x)
    counts_y = _Counter(a.tobytes() for a in array_y)

    return len(counts_y - counts_x) == 0


def within(array_x, array_y):
    """Check if the elements of array_x are a sub-set 
    of the elements of array_y.

    Parameters
    ----------
    array_x, array_y : ndarray, int
        Input arrays to be treated as multi-sets.

    Returns
    -------
    subset : bool
        `True` if elements of array_x are a sub-set
        of the elements of array_y.

    Notes
    -----
    Input arrays are treated as multi-sets in which
    the outer dimention is as an un-ordered collection 
    of elements.
    
    """
    counts_x = _Counter(a.tobytes() for a in array_x)
    counts_y = _Counter(a.tobytes() for a in array_y)

    return len(counts_x - counts_y) == 0


def unique_idx(array):
    """Return the index of the first occurance of each
    unique element within the outer dimention of an array.

    Parameters
    ----------
    array : ndarray, int
        Array of elements which may be sub-arrays.

    Returns
    -------
    index_array : ndarray, int, shape (unique_elements, )
        The index of the first occurrence of each unique 
        element.

    """
    strings = {a.tobytes() for a in array}
    idx = np.zeros(len(array)).astype(bool)
    for i in range(len(idx)):
        string = array[i].tobytes()
        if string in strings:
            strings -= {string}
            idx[i] = True
    return idx


def unique(array):
    """Return the unique elements within the outer 
    dimention of an array.

    Parameters
    ----------
    array : ndarray, int
        Array of elements which may be sub-arrays.

    Returns
    -------
    unique_array : ndarray, int
        The unique elements of the
        input array.

    """
    return array[unique_idx(array)]


def categorize(array, categories):
    """Label the elements of an array using the
    elements of a second array as categories.

    Parameters
    ----------
    array : ndarray, int
        Input array of elements which may be sub-arrays.
    categories : ndarray, int
        Array with elements which are of the same dimentionality
        and type as those of the first input array.
    
    Returns
    -------
    labels_array : ndarray, int, shape (n_categories, )
        The index of the first occurrence of each element
        of the input array in the categories array. 

    Notes
    -----
    Elements of the input array that are not found within
    the category array will result in a label of `-1`.

    """
    assert categories.ndim == array.ndim
    assert categories.dtype == array.dtype
    # category indices are category labels
    labels = {}
    for i, cat in enumerate(categories):
        labels[cat.tobytes()] = i
    labeled = np.empty(len(array), int)
    for i, a in enumerate(array):
        label = labels.get(a.tobytes(), -1)  # -1 is unlabeled
        labeled[i] = label
    return labeled


def count(array, categories):
    """Count the occurance of each element of a
    category array within an input array.

    Parameters
    ----------
    array : ndarray, int
        Input array of elements which may be sub-arrays.
    categories : ndarray, int
        Array with elements which are of the same dimentionality
        and type as those of the first input array.
    
    Returns
    -------
    counts_array : ndarray, int, shape (n_elements, )
        The count of each element of the categories array
        that is found within the first input array.

    Notes
    -----
    The counts within the counts_array are returned in 
    the same order as elements of the categories array.

    """
    assert categories.ndim == array.ndim
    assert categories.dtype == array.dtype
    strings = _Counter(a.tobytes() for a in array)
    counts = np.zeros(len(categories), dtype=int)
    for i, cat in enumerate(categories):
        string = cat.tobytes()
        if string in strings:
            counts[i] = strings[string]
        else:
            counts[i] = 0
    return counts


def unique_counts(array, order=None):
    """Count the unique elements of an array where the elements
    may be sub-arrays.

    Parameters
    ----------
    array : ndarray, int
        Input array of elements which may be sub-arrays.
    order : str, optional
        Return results in 'ascending' or 'descending'
        order of counts.
    
    Returns
    -------
    unique_array : ndarray, int
        Unique elements of the outer dimention of the 
        input array.
    counts_array : ndarray, int, shape (n_elements, )
        The count of each unique element.

    """
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
    """Repeat each element of an array a specified number
    of time.

    Parameters
    ----------
    array : ndarray, int
        Input array of elements which may be sub-arrays.
    counts : ndarray, int, shape (n_elements, )
        The number of times to replicate each element.

    output_array : ndarray, int
        Array with elements repeated the specified number
        of times.

    Notes
    -----
    A repeat number of `0` will remove that element from
    the output array.

    """
    assert len(array) == len(counts)
    idx = np.repeat(np.arange(len(counts)), counts)
    return array[idx]
