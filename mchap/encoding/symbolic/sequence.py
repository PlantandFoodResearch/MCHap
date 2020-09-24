import numpy as np


def is_gap(array, gap='-'):
    """Identify gap positions in an character encoded biological sequence.

    Parameters
    ----------
    array : ndarray, str
        Array of characters encoding alleles.
    gap : str, optional
        Symbol used to encode gaps in the sequence.

    Returns
    -------
    mask : ndarray, bool
        Array of booleans indicating gap positions.

    """
    return array == gap


def depth(array, gap='-'):
    """Position-wise depth of a set of biological sequences that are encoded as characters.

    Parameters
    ----------
    array : ndarray, str, shape (n_sequences, n_positions)
        2D array of characters encoding a series of biological sequences.
    gap : str, optional
        Symbol used to encode gaps in the sequence.

    Returns
    -------
    depth : ndarray, int, shape (n_positions, )
        1D array of integer depth per position.

    Notes
    -----
    Gap values do not count towards depth.

    """
    return np.sum(array != gap, axis=0)

