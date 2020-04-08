import numpy as np

from haplohelper import mset
from haplohelper.encoding import allelic


def unique_kmer_freq(read_calls, haplotype_calls, k=3):
    """ Calculates position-wise frequency of kmers present in read_calls
    but absent in haplotype_calls.
    """
    # create kmers and counts
    read_kmers, read_kmer_counts = allelic.kmer_counts(read_calls, k=k)
    hap_kmers, _ = allelic.kmer_counts(haplotype_calls, k=k)
    
    # index of kmers not found in haplotypes
    idx = mset.count(hap_kmers, read_kmers).astype(np.bool) == False
    
    # depth of unique kmers
    unique_depth = allelic.depth(read_kmers[idx], read_kmer_counts[idx])

    # depth of total kmers
    depth = allelic.depth(read_kmers, read_kmer_counts)

    # return 
    return unique_depth / depth


def kmer_filter(read_calls, haplotype_calls, k=3, threshold=0.05):
    freqs = unique_kmer_freq(read_calls, haplotype_calls, k=k)
    return np.any(freqs > threshold)


def depth_filter(read_calls, threshold=5):
    depth = allelic.depth(read_calls)
    return np.any(depth < threshold)


def probability_filter(probability, threshold=0.95):
    return probability < threshold