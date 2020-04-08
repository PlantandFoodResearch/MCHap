#!/usr/bin/env python3

import os
import pysam
import numpy as np

from haplohelper.io import util


def extract_read_calls(locus, path, sample='ID', min_quality=20, read_dicts=False, set_sequence=False):
    """Read variants defined for a locus from an alignment file
    
    Sample is used as the sample identifier and may refere to any field in 
    a reads read-group e.g. 'ID' or 'SM'.
    If sample is specified as None then the bam file name is used
    """
    
    assert sample in {'ID', 'SM', None}
    
    n_positions = len(locus.positions)
    
    # dtype for allele call
    dtype = np.dtype([('allele', np.int8), ('qual', np.uint8)])
    
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
    
    # check for duplicate samples
    if sample is 'ID':
        # check no sample ID is reused from another bam
        for dictionary in bam.header['RG']:
            sample_key = dictionary['ID']
            if sample_key in data:
                raise IOError('Duplicate read group ID: {}'.format(sample_key))
            data[sample_key] = {}
    
    elif sample is 'SM':
        # store sample ids in dict for easy access
        sample_keys = {}
        # check no sample SM is reused from another bam
        for dictionary in bam.header['RG']:
            sample_key = dictionary['SM']
            if sample_key in data:
                raise IOError('Duplicate read group SM: {}'.format(sample_key))
            data[sample_key] = {}
            # map read group ID to SM field
            sample_keys[dictionary['ID']] = dictionary['SM']
            
    else:
        # if sample is None then use bam name as sample id
        sample_key = os.path.basename(bam.filename.decode())
        if sample_key in data:
            raise IOError('Duplicate file name: {}'.format(sample_key))
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
            
            # sample identifier
            if sample is 'ID':
                sample_key = read.get_tag('RG')
            elif sample is 'SM':
                sample_key = sample_keys[read.get_tag('RG')]
            else:
                # sample key defined above
                pass

            # get sample specific reads
            sample_data = data[sample_key]

            if read.qname not in sample_data:
                # default is array of gaps with qual of 0
                array = np.zeros(n_positions, dtype=dtype)
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
        for sample, reads in data.items():
            data[sample] = np.array(list(reads.values()))

    return data


def read_sample_ids(bam_paths, id='ID'):
   
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
