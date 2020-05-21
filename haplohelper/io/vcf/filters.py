import numpy as np
from dataclasses import dataclass

from haplohelper import mset
from haplohelper.encoding import allelic, symbolic


_PASS_CODE = 'PASS'
_NULL_CODE = '.'
_KMER_CODE = 'k{k}<{threshold}'
_DEPTH_CODE = 'd<{threshold}'
_PROB_CODE = 'p<{threshold}'
_FILTER_HEADER = '##FILTER=<ID={code},Description="{desc}">\n'


@dataclass(frozen=True)
class FilterHeader(object):
    id: str
    descr: str

    def header(self):
        template = '##FILTER=<ID={id},Description="{descr}">'
        return template.format(
            id=self.id,
            descr=self.descr
        )


@dataclass(frozen=True)
class FilterCall(object):
    id: str
    failed: bool
    applied: bool = True
    
    def __str__(self):
        if self.applied:
            return self.id if self.failed else 'PASS'
        else: 
            return '.'


@dataclass(frozen=True)
class FilterCallSet(object):
    calls: tuple
        
    def __str__(self):
        calls = [call for call in self.calls if call.applied]

        if len(calls) == 0:
            return '.'
        else:
            failed = [call for call in calls if call.failed]
            
            if failed:
                return ','.join(map(str, failed))
            else:
                return 'PASS'


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


def kmer_filter_header(k=3, threshold=0.95):
    code = _KMER_CODE.format(k=k, threshold=threshold)
    descr = 'Less than {} % of samples read-variant {}-mers '.format(threshold * 100, k)
    return FilterHeader(code, descr)


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
    code = _KMER_CODE.format(k=k, threshold=threshold)

    n_pos = variants.shape[-1]
    if n_pos < k:
        # can't apply kmer filter
        return FilterCall(code, None, applied=False)

    freqs = kmer_representation(variants, genotype, k=k)
    fail = np.any(freqs < threshold)

    return FilterCall(code, fail)


def depth_filter_header(threshold=5.0):
    code = _DEPTH_CODE.format(threshold=threshold)
    descr = 'Sample has mean read depth less than {}.'.format(threshold)
    return FilterHeader(code, descr)


def depth_variant_filter(depths, threshold=5.0, gap='-'):
    fails = depths < threshold
    code = _DEPTH_CODE.format(threshold=threshold)
    return [code if fail else _PASS_CODE for fail in fails]


def depth_haplotype_filter(depths, threshold=5.0, gap='-'):
    code = _DEPTH_CODE.format(threshold=threshold)
    fail = np.mean(depths) < threshold
    return FilterCall(code, fail)


def prob_filter_header(threshold=0.95):
    descr = 'Samples genotype posterior probability < {}.'.format(threshold)
    code = 'p<{}'.format(threshold)
    return FilterHeader(code, descr)


def prob_filter(p, threshold=0.95):
    fails = p < threshold
    code = 'p<{}'.format(threshold)
    if np.shape(p) == ():
        # scalar
        return FilterCall(code, fails)
    else:
        # iterable
        return [code if fail else _PASS_CODE for fail in fails]
