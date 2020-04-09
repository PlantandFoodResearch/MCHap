import numpy as np

from haplohelper import mset
from haplohelper.encoding import allelic
from haplohelper.io.bam import dtype_allele_call as _dtype_allele_call


def kmer_representation(read_calls, haplotype_calls, k=3):
    """ Calculates position-wise frequency of read_calls kmers which 
    are also present in haplotype_calls.
    """
    if read_calls.dtype == _dtype_allele_call:
        # we only need the allele calls not the quals
        read_calls = read_calls['allele']

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
    return 1 - (unique_depth / depth)



class Filter(object):

    _header = '##FILTER=<ID={},Description="{}">\n'

    def __call__(self):
        """Returns True if filter fails
        """
        raise NotImplementedError()

    @property
    def code(self):
        """Returns VCF filter code described in description
        """
        raise NotImplementedError()

    @property
    def header(self):
        """Returns VCF header description string
        """
        raise NotImplementedError()


class KmerFilter(Filter):

    def __init__(self, k=3, threshold=0.95):
        self.k = k
        self.threshold = threshold

    def __call__(self, read_calls, haplotype_calls, return_code=True):
        if read_calls.dtype == _dtype_allele_call:
            # we only need the allele calls not the quals
            read_calls = read_calls['allele']
        freqs = kmer_representation(read_calls, haplotype_calls, k=self.k)

        fail = np.any(freqs < self.threshold)

        if return_code:
            return self.code if fail else 'PASS'
        else:
            return fail

    @property
    def code(self):
        return 'k{}<{}'.format(self.k, self.threshold)

    @property
    def header(self):
        description = 'Less than {} % of samples read-variant {}-mers '.format(
            self.threshold * 100,
            self.k
        )
        description += 'present in called haplotypes at one or more positions.'
        return self._header.format(self.code, description)


class DepthFilter(Filter):

    def __init__(self, threshold=5, reduce='mean'):

        self.reduce = reduce
        self.threshold = threshold

    def __call__(self, read_calls, return_code=True):

        if read_calls.dtype == _dtype_allele_call:
            # we only need the allele calls not the quals
            read_calls = read_calls['allele']

        depth = allelic.depth(read_calls)

        if self.reduce is 'mean':
            depth = np.mean(depth)
        elif self.reduce is 'max':
            depth = np.max(depth)
        elif self.reduce is 'min':
            depth = np.min(depth)
        else:
            raise ValueError('"{}" is not a recognised reduction'.format(self.reduce))

        fail = depth < self.threshold
        if return_code:
            return self.code if fail else 'PASS'
        else:
            return fail

    @property
    def code(self):
        return 'd<{}'.format(self.threshold)

    @property
    def header(self):
        description = 'Sample has {} read depth less than {}.'.format(self.reduce, self.threshold)
        return self._header.format(self.code, description)


class ProbabilityFilter(Filter):

    def __init__(self, threshold=0.95):

        self.threshold = threshold

    def __call__(self, probability, return_code=True):
        fail = probability < self.threshold
        if return_code:
            return self.code if fail else 'PASS'
        else:
            return fail

    @property
    def code(self):
        return 'p<{}'.format(self.threshold)

    @property
    def header(self):
        description = 'Samples genotype posterior probability < {}.'.format(self.threshold)
        return self._header.format(self.code, description)


def combine_codes(*args):
    string = ';'.join(code for code in args if code != 'PASS')
    if string:
        return string
    else:
        return 'PASS'
