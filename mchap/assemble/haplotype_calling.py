import numpy as np


def call_posterior_haplotypes(posteriors, threshold=0.01):
    """Call haplotype alleles for VCF output from a population
    of genotype posterior distributions.

    Parameters
    ----------
    posteriors : list, PosteriorGenotypeDistribution
        A list of individual genotype posteriors.
    threshold : float
        Minimum required posterior probability of occurrence
        with in any individual for a haplotype to be included.

    Returns
    -------
    haplotypes : ndarray, int, shape, (n_haplotypes, n_base)
        VCF sorted haplotype arrays.
    ref_observed : bool
        Bool indicating that the reference allele was called.
    """
    # maps of bytes to arrays and bytes to sum probs
    haplotype_arrays = {}
    haplotype_values = {}
    # iterate through genotype posterors
    for post in posteriors:
        # include haps based on probability of occurrence
        (
            haps,
            probs,
        ) = post.allele_occurrence()
        _, weights = post.allele_frequencies(dosage=True)
        idx = probs >= threshold
        # order haps based on weighted prob
        haps = haps[idx]
        weights = weights[idx]
        for h, w in zip(haps, weights):
            b = h.tobytes()
            if b not in haplotype_arrays:
                haplotype_arrays[b] = h
                haplotype_values[b] = 0
            haplotype_values[b] += w
    # remove reference allele if present
    refbytes = None
    for b, h in haplotype_arrays.items():
        if np.all(h == 0):
            # ref allele
            refbytes = b
    if refbytes is not None:
        haplotype_arrays.pop(refbytes)
        haplotype_values.pop(refbytes)
        ref_observed = True
    else:
        ref_observed = False
    # combine all called haplotypes into array
    n_alleles = len(haplotype_arrays) + 1
    n_base = posteriors[0].genotypes.shape[-1]
    haplotypes = np.full((n_alleles, n_base), -1, np.int8)
    values = np.full(n_alleles, -1, float)
    for i, (b, h) in enumerate(haplotype_arrays.items()):
        p = haplotype_values[b]
        haplotypes[i] = h
        values[i] = p
    haplotypes[-1][:] = 0  # ref allele
    values[-1] = values.max() + 1
    order = np.flip(np.argsort(values))
    return haplotypes[order], ref_observed
