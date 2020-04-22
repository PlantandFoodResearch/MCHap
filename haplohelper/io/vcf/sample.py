import numpy as np
from dataclasses import dataclass

from haplohelper import mset
from haplohelper.encoding import allelic, symbolic

_PASS_CODE = 'PASS'
_NULL_CODE = '.'
_KMER_CODE = 'k{k}<{threshold}'
_DEPTH_CODE = 'd<{threshold}'
_PROB_CODE = 'p<{threshold}'


def kmer_representation(variants, haplotype_calls, k=3):
    """ Calculates position-wise frequency of read_calls kmers which 
    are also present in haplotype_calls.
    """
    # create kmers and counts
    read_kmers, read_kmer_counts = allelic.kmer_counts(variants, k=k)
    hap_kmers, _ = allelic.kmer_counts(haplotype_calls, k=k)
    
    # index of kmers not found in haplotypes
    idx = mset.count(hap_kmers, read_kmers).astype(np.bool) == False
    
    # depth of unique kmers
    unique_depth = allelic.depth(read_kmers[idx], read_kmer_counts[idx])

    # depth of total kmers
    depth = allelic.depth(read_kmers, read_kmer_counts)

    # avoid divide by zero 
    with np.errstate(divide='ignore',invalid='ignore'):
        result = 1 - np.where(depth > 0, unique_depth / depth, 0)

    return result


def kmer_variant_filter(variants, genotype, k=3, threshold=0.95):
    n_pos = variants.shape[-1]
    if n_pos < k:
        # can't apply kmer filter
        return [_NULL_CODE for _ in range(n_pos)]

    freqs = kmer_representation(variants, genotype, k=k)
    fails = freqs < threshold
    code = _KMER_CODE.format(k=k, threshold=threshold)
    return [code if fail else _PASS_CODE for fail in fails]


def kmer_haplotype_filter(variants, genotype, k=3, threshold=0.95):
    n_pos = variants.shape[-1]
    if n_pos < k:
        # can't apply kmer filter
        return _NULL_CODE

    freqs = kmer_representation(variants, genotype, k=k)
    fail = np.any(freqs < threshold)
    if fail:
        return _KMER_CODE.format(k=k, threshold=threshold)
    else:
        return _PASS_CODE


def depth_variant_filter(depths, threshold=5.0, gap='-'):
    fails = depths < threshold
    code = _DEPTH_CODE.format(threshold=threshold)
    return [code if fail else _PASS_CODE for fail in fails]


def depth_haplotype_filter(depths, threshold=5.0, gap='-'):
    fail = np.mean(depths) < threshold
    if fail:
        return _DEPTH_CODE.format(threshold=threshold)
    else:
        return _PASS_CODE


def prob_filter(p, threshold=0.95):
    fails = p < threshold
    code = 'p<{}'.format(threshold)
    if np.shape(p) == ():
        # scalar
        if fails:
            return code
        else:
            return _PASS_CODE
    else:
        # iterable
        return [code if fail else _PASS_CODE for fail in fails]


def combine_filters(*args):
    if np.ndim(args) == 1:

        args = [arg for arg in args if arg != _NULL_CODE]
        if len(args) == 0:
            # no filters applied
            return _NULL_CODE
        string = ';'.join((arg for arg in args if arg != _PASS_CODE))
        if string:
            return string
        else:
            return _PASS_CODE
    else:
        return [combine_filters(arg) for arg in np.transpose(args)]
