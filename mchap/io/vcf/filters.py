import numpy as np
from dataclasses import dataclass

from mchap import mset
from mchap.encoding import allelic, symbolic


@dataclass(frozen=True)
class SampleFilter(object):

    def __str__(self):
        template = '##FILTER=<ID={id},Description="{descr}">'
        return template.format(
            id=self.id,
            descr=self.descr
        )

    def __call__(self):
        raise NotImplementedError

    @property
    def id(self):
        raise NotImplementedError

    @property
    def descr(self):
        raise NotImplementedError


@dataclass(frozen=True)
class SamplePassFilter(SampleFilter):
    @property
    def id(self):
        return 'PASS'

    @property
    def descr(self):
        return 'All filters passed'


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

    @property
    def failed(self):
        for call in self.calls:
            if call.applied and call.failed:
                return True
        return False


def _kmer_representation(variants, haplotype_calls, k=3):
    """ Calculates position-wise frequency of read_calls kmers which 
    are also present in haplotype_calls.
    """
    # create kmers and counts
    read_kmers, read_kmer_counts = allelic.kmer_counts(variants, k=k)
    hap_kmers, _ = allelic.kmer_counts(haplotype_calls, k=k)
    
    # handle case of no read kmers (i.e. from nan reads)
    if np.prod(read_kmers.shape) == 0:
        _, n_pos = hap_kmers.shape
        return np.ones(n_pos)

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


@dataclass(frozen=True)
class SampleKmerFilter(SampleFilter):
    k: int = 3
    threshold: float = 0.95
        
    @property
    def id(self):
        return '{}m{}'.format(self.k, int(self.threshold * 100))
    
    @property
    def descr(self):
        template = 'Less than {} percent of read-variant {}-mers represented in haplotypes'
        return template.format(self.threshold * 100, self.k)
    
    def __call__(self, variants, genotype):
        n_pos = variants.shape[-1]
        if n_pos < self.k:
            # can't apply kmer filter
            return FilterCall(self.id, None, applied=False)

        freqs = _kmer_representation(variants, genotype, k=self.k)
        fail = np.any(freqs < self.threshold)

        return FilterCall(self.id, fail)


@dataclass(frozen=True)
class SampleDepthFilter(SampleFilter):
    threshold: float = 5.0
        
    @property
    def id(self):
        return 'dp{}'.format(int(self.threshold))
    
    @property
    def descr(self):
        template = 'Sample has mean read depth less than {}'
        return template.format(self.threshold)
    
    def __call__(self, depths, gap='-'):
        if np.prod(depths.shape) == 0:
            # can't apply depth filter across 0 variants
            return FilterCall(self.id, None, applied=False)
        fail = np.mean(depths) < self.threshold
        return FilterCall(self.id, fail)


@dataclass(frozen=True)
class SampleReadCountFilter(SampleFilter):
    threshold: int = 5
        
    @property
    def id(self):
        return 'rc{}'.format(int(self.threshold))
    
    @property
    def descr(self):
        template = 'Sample has read (pair) count of less than {}'
        return template.format(self.threshold)
    
    def __call__(self, count):
        fails = count < self.threshold
        return FilterCall(self.id, fails)


@dataclass(frozen=True)
class SamplePhenotypeProbabilityFilter(SampleFilter):
    threshold: float = 0.95
        
    @property
    def id(self):
        return 'pp{}'.format(int(self.threshold * 100))
    
    @property
    def descr(self):
        template = 'Samples phenotype posterior probability less than {}'
        return template.format(self.threshold)
    
    def __call__(self, p):
        fails = p < self.threshold
        return FilterCall(self.id, fails)
