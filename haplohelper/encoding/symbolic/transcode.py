import numpy as np


def as_allelic(array, alleles=None, dtype=np.int8):

    if not isinstance(array, np.ndarray):
        array = np.array(array, copy=False)
    
    if np.ndim(array) == 1:
        n_seq, n_pos = 1, len(array)
    else:
        n_seq, n_pos = array.shape[-2:]
        
    symbols = array.reshape(n_seq, n_pos)
        
    if alleles is None:
        # assume non-gap symbols are integers
        d = {s: int(s) for s in np.unique(symbols) if s.isdigit()}
        alleles = [d] * n_pos
    else:
        alleles = [{k: v for v, k in enumerate(tup)} for tup in alleles]
        
    new = np.empty(symbols.shape, dtype=dtype)
    
    for j in range(n_seq):
        for i in range(n_pos):
            s = symbols[j, i]
            a = alleles[i].get(s, -1)  # gap if not recognised
            new[j, i] = a
    
    return new.reshape(array.shape)


