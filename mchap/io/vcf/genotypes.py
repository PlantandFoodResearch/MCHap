import numpy as np
from collections import Counter


def sort_haplotypes(genotypes, dtype=np.int8):
    """Sort unique haplotypes from multiple genotype arrays from
    most to least frequent with the reference allele first

    Parameters
    ----------
    genotypes : list
        List of ndarrays each with shape (ploidy, n_positions).
    dtype : type
        Numpy dtype for returned array.
    
    Returns
    -------
    haplotypes : ndarray, int, shape (n_haplotypes, n_positions)
        Unique haplotypes sorted by frequency with reference allele first.
    counts : ndarray, int, shape (n_haplotypes, )
        Count of each haplotype

    """
    haplotypes = np.concatenate(genotypes)
    _, n_pos = haplotypes.shape
    
    # count observed haps
    counts = Counter(tuple(hap) for hap in haplotypes)
    
    # ref and null haps are special values
    ref = (0, ) * n_pos
    null = (-1, ) * n_pos
    
    # remove null haps from count if present
    if null in counts:
        _ = counts.pop(null)
    
    # seperate ref count
    if ref not in counts:
        ref_count = 0
    else:
        ref_count = counts.pop(ref)

    # order by frequency then insert ref first
    pairs = counts.most_common()
    pairs = [(ref, ref_count)] + pairs
    
    # convert back to arrays
    haplotypes = np.array([a for a, _ in pairs], dtype=dtype)
    counts = np.array([c for _, c in pairs])
    
    return haplotypes, counts


def genotype_string(genotype, haplotypes):
    """Convert a genotype array to a VCF genotype (GT) string

    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_positions)
        Integer encoded genotype.
    haplotypes : ndarray, int, shape (n_haplotypes, n_positions)
        Unique haplotypes sorted by frequency with reference allele first.

    Returns
    -------
    string : str
        VCF genotype string.
    """
    assert genotype.dtype == haplotypes.dtype
    labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}
    alleles = [labels.get(h.tobytes(), -1) for h in genotype]
    alleles.sort()
    chars = [str(a) for a in alleles if a >= 0]
    chars += ['.' for a in alleles if a < 0]
    return '/'.join(chars)


def expected_dosage(genotypes, probabilities, haplotypes):
    """Calculate the expected floating point dosage based on a phenotype distribution.

    Parameters
    ----------
    genotypes : ndarray, int, shape (n_genotypes, ploidy, n_positions)
        Integer encoded genotypes containing the same alleles in different dosage.
    probabilities : ndarray, int, shape (n_genotypes, ).
        Probability of each genotype.

    Returns
    -------
    expected_dosage: ndarray, float, shape (n_alleles, )
        Expected count of each allele.
    """
    assert genotypes.dtype == haplotypes.dtype

    # if all alleles are null then no dosage
    if np.all(genotypes < 0):
        return np.array([np.nan])

    # label genotype alleles
    labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}
    alleles = [[labels[h.tobytes()] for h in g] for g in genotypes]

    # values for sorting genotypes, this works because all
    # genotypes share the same alleles in different dosage
    vals = [np.sum(g) for g in alleles]
    
    # normalised probability of each dosage
    probs = probabilities[np.argsort(vals)]
    probs /= probs.sum()
    
    # per genotype allele counts
    uniques, counts = zip(*[np.unique(g, return_counts=True) for g in alleles])
    counts = np.array(counts)
    
    # assert all dosage variants contain same alleles
    for i in range(1, len(uniques)):
        np.testing.assert_array_equal(uniques[0], uniques[i])
    
    # expectation of dosage
    expected = np.sum(counts * probs[:, None], axis=0)
    
    return expected
