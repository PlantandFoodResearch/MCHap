#!/usr/bin/env python3

import os
import pysam
import numpy as np

from haplohelper.io import util
from haplohelper.encoding.allelic import as_probabilistic as _as_probabilistic
from haplohelper.encoding.symbolic import as_allelic as _as_allelic


def extract_sample_ids(bam_paths, id='ID'):
   
    if id is None:
        data = {os.path.basename(path): path for path in bam_paths}
    
    else:
        data = {}
        
        for path in bam_paths:
            bam = pysam.AlignmentFile(path)
            sample_names = [read_group[id] for read_group in bam.header['RG']]
            for sample in sample_names:
                if sample in data:
                    raise IOError('Duplicate sample with id = "{}" in file "{}"'.format(sample, path))
                else:
                    data[sample] = path
    return data


def extract_read_variants(
        locus, 
        path, 
        samples=None,
        id='ID',
        min_quality=20, 
        read_dicts=False, 
    ):
    """Read variants defined for a locus from an alignment file
    
    Sample is used as the sample identifier and may refere to any field in 
    a reads read-group e.g. 'ID' or 'SM'.
    """
    
    assert id in {'ID', 'SM'}

    # a single sample name may be given as a string
    if isinstance(samples, str):
        samples = {samples}
    
    n_positions = len(locus.positions)
    
    # create mapping of ref position to variant index
    positions = {pos: i for i, pos in enumerate(locus.positions)}
    
    data={}

    bam = pysam.AlignmentFile(path)
    
    # store sample ids in dict for easy access
    # sample_keys is a map of RG ID to ID or SM
    sample_keys = {}
    for dictionary in bam.header['RG']:

        # sample key based on a user defined readgroup field
        sample_key = dictionary[id]

        # map read group ID to RG field (may be ID to ID)
        sample_keys[dictionary['ID']] = sample_key

        # check no sample id is reused from another bam
        if sample_key in data:
            raise IOError('Duplicate sample id: {}'.format(sample_key))
        # only add specified samples to returned dict
        elif samples and sample_key not in samples:
            # this read is not from a sample in the specified set
            pass
        else:
            # add sample to the dict
            data[sample_key] = {}

    # iterate through reads
    reads = bam.fetch(locus.contig, locus.start, locus.stop)
    for read in reads:
        
        if read.is_unmapped:
            # skip read
            pass
        
        elif read.mapping_quality < min_quality:
            # skip read
            pass
        
        else:
            
            # look up sample identifier based on RG ID
            sample_key = sample_keys[read.get_tag('RG')]

            if samples and sample_key not in samples:
                # this read is not from a sample in the specified set
                pass

            else:
                # get sample specific reads
                sample_data = data[sample_key]

                if read.qname not in sample_data:
                    # default is array of gaps with qual of 0
                    chars = np.empty(n_positions, dtype='U1')
                    chars[:] = '-'
                    quals = np.zeros(n_positions, dtype=np.int8)
                    sample_data[read.qname] = [chars, quals]
                    
                else:
                    # reuse arrays for first read in pair
                    chars, quals = sample_data[read.qname]

                for read_pos, ref_pos, ref_char in read.get_aligned_pairs(matches_only=True, with_seq=True):

                    # if this is a variant position then extract the call and qual
                    if ref_pos in positions:
                        idx = positions[ref_pos]

                        # references allele in bam should match reference allele in locus
                        assert locus.alleles[idx][0].upper() == ref_char.upper()
                                                
                        char = read.seq[read_pos]
                        qual = util.qual_of_char(read.qual[read_pos])
                        
                        if  chars[idx] == '-':
                            # first observation
                            chars[idx] = char
                            quals[idx] = qual
                        
                        elif chars[idx] == char:
                            # second call is congruent
                             quals[idx] += qual
                                
                        else:
                            # incongruent calls
                            chars[idx] = 'N'
                            quals[idx] += 0
    
    if read_dicts:
        # return a dict of dicts of arrays
        pass
    
    else:
        # return a dict of matrices
        for id, reads in data.items():
            tuples = list(reads.values())
            chars = np.array([tup[0] for tup in tuples])
            quals = np.array([tup[1] for tup in tuples])
            data[id] = (chars, quals)

    return data


def encode_read_alleles(locus, symbols):
    if np.size(symbols) == 0:
        # This is a hack to let static computation graphs compleate 
        # when there are no reads for a sample
        # by specifying a single read of gaps only, the posterior
        # should approximate the prior
        symbols = np.array(['-'] * len(locus.variants))

    return _as_allelic(symbols, alleles=locus.alleles)


def encode_read_distributions(locus, calls, quals, error_rate=util.PFEIFFER_ERROR, gaps=True):

    # convert error_rate to prob of correct call
    probs = np.ones((calls.shape), dtype=np.float) * (1 - error_rate)

    # convert qual scores to probs and multiply
    probs *= util.prob_of_qual(quals)
    
    n_alleles = locus.count_alleles()
    encoded = _as_probabilistic(calls, n_alleles, probs, gaps=gaps)
    return encoded
