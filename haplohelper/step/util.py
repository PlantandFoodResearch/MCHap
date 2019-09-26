import numpy as np
from numba import jit

@jit(nopython=True)
def sum_log_prob(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    else:
        return y + np.log1p(np.exp(x - y))

@jit(nopython=True)
def rand_choice(arr, prob):
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@jit(nopython=True)
def array_equal(x, y):
    """Assumes equal length and dtype.
    """
    n = len(x)
    for i in range(n):
        if x[i] != y[i]:
            return False
    return True
