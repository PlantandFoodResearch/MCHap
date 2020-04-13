#!/usr/bin/env python3

import os
import pysam
import numpy as np

from haplohelper.io import util
from haplohelper.encoding.allelic import as_probabilistic as _as_probabilistic


# dtype for allele call with qual score
dtype_allele_call = np.dtype([('allele', np.int8), ('qual', np.uint8)])


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


def extract_read_calls(
        locus, 
        path, 
        samples=None,
        rg_field='ID',
        min_quality=20, 
        read_dicts=False, 
        set_sequence=False  # TODO: remove this
    ):
    """Read variants defined for a locus from an alignment file
    
    Sample is used as the sample identifier and may refere to any field in 
    a reads read-group e.g. 'ID' or 'SM'.
    """
    
    assert rg_field in {'ID', 'SM'}

    # a single sample name may be given as a string
    if isinstance(samples, str):
        samples = {samples}
    
    n_positions = len(locus.positions)
    
    # create list of mappings of variants to allele integers
    encodings = [{string: i for i, string in enumerate(var)} for var in locus.alleles]
    
    # create mapping of ref position to variant index
    positions = {pos: i for i, pos in enumerate(locus.positions)}

    if set_sequence:
        # create an empty array to gather reference chars
        # use a set to keep track of remaining chars to get 
        ref_sequence = np.empty(locus.stop - locus.start, dtype = 'U1')
        ref_sequence[:] = 'N'  # default to unknown allele
        ref_sequence_remaining = set(locus.range)
    
    data={}

    bam = pysam.AlignmentFile(path)
    
    # store sample ids in dict for easy access
    # sample_keys is a map of RG ID to ID or SM
    sample_keys = {}
    for dictionary in bam.header['RG']:

        # sample key based on a user defined readgroup field
        sample_key = dictionary[rg_field]

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
                    array = np.zeros(n_positions, dtype=dtype_allele_call)
                    array['allele'] -= 1
                    sample_data[read.qname] = array
                    
                else:
                    # reuse array for first read in pair
                    array = sample_data[read.qname]

                for read_pos, ref_pos, ref_char in read.get_aligned_pairs(matches_only=True, with_seq=True):
                    
                    # check if (still) setting reference sequences
                    if set_sequence and ref_sequence_remaining:
                        # check if this pos still needs to be set
                        if ref_pos in ref_sequence_remaining:
                            # set character (may be lower case if read varies)
                            ref_sequence[ref_pos - locus.start] = ref_char.upper()
                            # remove position from remaining
                            ref_sequence_remaining.remove(ref_pos)

                    # if this is a variant position then extract the call and qual
                    if ref_pos in positions:
                        idx = positions[ref_pos]
                                                
                        char = read.seq[read_pos]
                        qual = util.qual_of_char(read.qual[read_pos])
                        
                        # Only proceed if char is a recognised allele.
                        # Note that if reads overlap and one is not a 
                        # recognised allele then it is ignored and other
                        # allele is used.
                        # If both are recognised but conflicting alleles
                        # then it's treated as a null/gap
                        # TODO: find a better way to handle this
                        if char in encodings[idx]:
                            allele = encodings[idx][char]
                        
                            if array[idx]['allele'] == -1:
                                # first observation of this position in this read pair
                                array[idx] = (allele, qual)
                            elif array[idx]['allele'] == allele:
                                # second observation of same allele
                                array[idx]['qual'] += qual  # combine qual
                            else:
                                # conflicting calls so treat as null
                                array[idx] = (-1, 0)

    # check if setting reference sequence
    if set_sequence:
        # check that all positions were recovered
        if ref_sequence_remaining:
            warning = 'Reference sequence not recoverd at positions {}.'
            warning += 'This is likely due to a read depth of 0.'
            Warning(warning.format(ref_sequence_remaining))
        
        # set the sequence
        locus.sequence = ''.join(ref_sequence)
    
    if read_dicts:
        # return a dict of dicts of arrays
        pass
    
    else:
        # return a dict of matrices
        for rg_field, reads in data.items():
            data[rg_field] = np.array(list(reads.values()))

    return data


def encode_read_calls(locus, read_calls, error_rate=util.PFEIFFER_ERROR, use_quals=True):
    alleles = read_calls['allele']

    # convert error_rate to prob of correct call
    probs = np.ones((alleles.shape), dtype=np.float) * (1 - error_rate)

    if use_quals:
        # convert qual scores to probs and multiply
        probs *= util.prob_of_qual(read_calls['qual'])
    
    n_alleles = locus.count_alleles()
    encoded = _as_probabilistic(alleles, n_alleles, probs, gaps=False)
    return encoded
