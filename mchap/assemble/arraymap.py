from numba import njit
import numpy as np


@njit(cache=True)
def new(array_length, node_branches, initial_size=32, max_size=2 ** 16):
    """Initialise a new array map for 1-dimensional integer arrays of fixed length.

    Parameters
    ----------
    array_length : int
        Length of arrays stored in this map.
    node_branches : int
        Number of possible branches from each node.
    initial_size : int
        Initial array length for nodes and values.
    max_size : int
        Maximum array length for nodes and values at which point storing
        new values will raise an error.

    Returns
    -------
    tree : np.array, int, shape (initial_size, node_branches)
        Array of array_map nodes in which -1 in a null value.
    values : nd.array, float, shape (initial_size, )
        Array of values stored in array_map.
    array_length : int
        Fixed size of arrays stored in this array_map.
    empty_node : int
        Index of the first empty node slot excluding 0.
    empty_value : int
        Index of the first empty values slot.
    max_size : int
        Maximum array length for nodes and values at which point storing
        new values will raise an error.

    Notes
    -----
    Returned values are not meant to be interacted with individually and should be
    treated as a single object.

    """
    assert initial_size >= 2
    tree = np.full((initial_size, node_branches), -1, np.int64)
    values = np.full(initial_size, np.nan, np.float64)
    return (tree, values, array_length, 1, 0, max_size)


@njit(cache=True)
def set(array_map, array, value, empty_if_full=False):
    """Set the value stored for a given array in an array_map.

    Parameters
    ----------
    array_map : tuple
        An array_map tuple.
    array : ndarray, int shape (array_length, )
        Integer 1-d array.
    value : float
        Value to store in array_map.
    empty_if_full : bool
        If true then the input array_map will be returned
        with all nodes and values removed instead of raising an
        error when trying to expand the array_map beyond
        its maximum size.

    Returns
    -------
    array_map : tuple
        An updated array_map tuple.

    Notes
    -----
    Elements of the array_map may be updated in place or replaced
    hence existing references to the array_map should not be reused.
    """
    if array_map is None:
        return array_map
    tree, values, array_length, empty_node, empty_values, max_size = array_map
    _, n_branches = tree.shape
    assert len(array) == array_length
    node = 0
    for i in range(len(array)):
        j = array[i]
        assert j < n_branches
        next_node = tree[node, j]
        if next_node < 0:
            # add a new node to the tree
            next_node = empty_node
            tree[node, j] = next_node
            empty_node += 1
            # expand tree array size if full
            if (empty_node + 1) >= len(tree):
                # tree is full so double size
                n_nodes, n_node_options = tree.shape
                if (n_nodes * 2) > max_size:
                    if empty_if_full:
                        # remove all nodes and values
                        tree[:] = -1
                        values[:] = np.nan
                        return (tree, values, array_length, 1, 0, max_size)
                    else:
                        raise ValueError(
                            "cannot expand array_map beyond its maximum size."
                        )
                new_tree = np.full((n_nodes * 2, n_node_options), -1, tree.dtype)
                new_tree[0:n_nodes] = tree
                tree = new_tree
        assert node < next_node < empty_node < len(tree)
        node = next_node
    # final node points to values
    value_idx = tree[node, 0]
    if value_idx < 0:
        value_idx = empty_values
        tree[node, 0] = value_idx
        empty_values += 1
        # expand values array size if full
        if (empty_values + 1) >= len(values):
            # values is full so double size
            n_values = len(values)
            if (n_values * 2) > max_size:
                if empty_if_full:
                    # remove all nodes and values
                    tree[:] = -1
                    values[:] = np.nan
                    return (tree, values, array_length, 1, 0, max_size)
                else:
                    raise ValueError("cannot expand array_map beyond its maximum size.")
            new_values = np.full((n_values * 2), np.nan, values.dtype)
            new_values[0:n_values] = values
            values = new_values
    values[value_idx] = value
    return (tree, values, array_length, empty_node, empty_values, max_size)


@njit(cache=True)
def get(array_map, array):
    """Retrive the value stored for a given array in an array_map.

    Parameters
    ----------
    array_map : tuple
        An array_map tuple.
    array : ndarray, int shape (array_length, )
        Integer 1-d array.

    Returns
    -------
    value : float
        Value stored in array_map.
    """
    if array_map is None:
        return np.nan
    tree, values, array_length, _, empty_values, _ = array_map
    assert len(array) == array_length
    node = 0
    for i in range(len(array)):
        j = array[i]
        next_node = tree[node, j]
        if next_node < 0:
            # no entry for this array
            return values[empty_values]
        node = next_node
    # final node points to values
    value_idx = tree[node, 0]
    if value_idx < 0:
        return values[empty_values]
    return values[value_idx]
