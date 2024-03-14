#!/usr/bin/env python3

import pysam
import numpy as np

from mchap.io import util
from mchap.encoding.integer import as_probabilistic as _as_probabilistic
from mchap.encoding.character import as_allelic as _as_allelic


__all__ = [
    "extract_sample_ids",
    "extract_read_variants",
    "encode_read_alleles",
    "encode_read_distributions",
]


ID_TAGS = {"ID", "SM"}


def extract_sample_ids(bam_paths, id="SM", reference_path=None):
    """Extract sample id's from a list of bam files.

    Parameters
    ----------
    bam_paths : list, str
        List of bam file paths.
    id : str
        Read-group field to use as sample id (default = 'SM').
        Must be one of "ID" or "SM".

    Returns
    -------
    sample_bams : dict, list, str
        Mapping of sample names a list of bam files.
    """
    assert id in ID_TAGS
    data = {}
    for path in bam_paths:
        bam = pysam.AlignmentFile(path, reference_filename=reference_path)
        # allow multiple read groups for a single sample within a bam file
        # but guard against duplicate sample identifier across multiple bams
        bam_data = {read_group[id]: path for read_group in bam.header["RG"]}
        for sample in bam_data.keys():
            if sample in data:
                raise IOError(
                    'Duplicate sample with id = "{}" in file "{}"'.format(sample, path)
                )
        data.update(bam_data)
    return data


def extract_read_variants(
    locus,
    alignment_file,
    samples=None,
    id="SM",
    min_quality=20,
    skip_duplicates=True,
    skip_qcfail=True,
    skip_supplementary=True,
    read_dicts=False,
):
    """Read variants defined for a locus from an alignment file

    Parameters
    ----------
    locus : Locus
        A locus object defining a genomic locus with known variants.
    alignment_file : AlignmentFile
        Instance of pysam.AlignmentFile.
    samples : list, str
        List of samples to extract from bam (default is to extract all samples).
    id : str
        Read group field to use as sample identifier: 'SM' or 'ID' (default = 'SM').
    min_quality : int
        Minimum allowed mapping quality of reads (default = 20).
    skip_duplicates : bool
        If true then reads marked as duplicates will be skipped.
    skip_qcfail : bool
        If true then reads marked as qcfail will be skipped.
    skip_supplementary : bool
        If true then reads marked as supplementary will be skipped.
    read_dicts : bool
        If true then a nested dictionary of samples to read-names to individual read arrays
        will be returned instead of a dictionary of samples to read matrices (used for testing).

    Returns
    -------
    sample_reads : dict, tuple, ndarray
        Mapping of sample names to a pair of matrices containing read base chars and quals.

    """

    assert id in ID_TAGS

    # a single sample name may be given as a string
    if isinstance(samples, str):
        samples = {samples}

    n_positions = len(locus.positions)

    # create mapping of ref position to variant index
    positions = {pos: i for i, pos in enumerate(locus.positions)}

    data = {}

    # store sample ids in dict for easy access
    # sample_keys is a map of RG ID to ID or SM
    sample_keys = {}
    for dictionary in alignment_file.header["RG"]:

        # sample key based on a user defined readgroup field
        sample_key = dictionary[id]

        # map read group ID to RG field (may be ID to ID)
        sample_keys[dictionary["ID"]] = sample_key

        # only add specified samples to returned dict
        if samples and sample_key not in samples:
            # this read is not from a sample in the specified set
            pass
        else:
            # add sample to the dict
            data[sample_key] = {}

    # iterate through reads
    reads = alignment_file.fetch(locus.contig, locus.start, locus.stop)
    for read in reads:

        if read.is_unmapped:
            # skip read
            pass

        elif read.mapping_quality < min_quality:
            # skip read
            pass

        elif read.is_duplicate and skip_duplicates:
            # skip read
            pass

        elif read.is_qcfail and skip_qcfail:
            # skip read
            pass

        elif read.is_supplementary and skip_supplementary:
            # skip read
            pass

        else:

            # look up sample identifier based on RG ID
            sample_key = sample_keys[read.get_tag("RG")]

            if samples and sample_key not in samples:
                # this read is not from a sample in the specified set
                pass

            else:
                # get sample specific reads
                sample_data = data[sample_key]

                if read.qname not in sample_data:
                    # default is array of gaps with qual of 0
                    chars = np.empty(n_positions, dtype="U1")
                    chars[:] = "-"
                    quals = np.zeros(n_positions, dtype=np.int16)
                    sample_data[read.qname] = [chars, quals]

                else:
                    # reuse arrays for first read in pair
                    chars, quals = sample_data[read.qname]

                for read_pos, ref_pos, ref_char in read.get_aligned_pairs(
                    matches_only=True, with_seq=True
                ):

                    # if this is a variant position then extract the call and qual
                    if ref_pos in positions:
                        idx = positions[ref_pos]

                        # references allele in bam should match reference allele in locus
                        if locus.alleles[idx][0].upper() != ref_char.upper():
                            path = alignment_file.filename.decode()
                            locus_ref_char = locus.alleles[idx][0]
                            locus_contig = locus.contig
                            locus_name = locus.name
                            vcf_pos = ref_pos + 1
                            if locus_name:
                                loc = f"'{locus_contig}:{vcf_pos}' in target '{locus_name}'"
                            else:
                                loc = f"'{locus_contig}:{vcf_pos}'"
                            raise ValueError(
                                f"Reference allele of variant '{locus_ref_char}' does not match alignment reference allele '{ref_char}' at position {loc} in '{path}'"
                            )

                        char = read.seq[read_pos]
                        qual = util.qual_of_char(read.qual[read_pos])

                        if chars[idx] == "-":
                            # first observation
                            chars[idx] = char
                            quals[idx] = qual

                        elif chars[idx] == char:
                            # second call is congruent
                            quals[idx] += qual

                        else:
                            # incongruent calls
                            chars[idx] = "N"
                            quals[idx] += 0

    if read_dicts:
        # return a dict of dicts of arrays
        pass

    else:
        # return a dict of matrices
        for id, reads in data.items():
            tuples = list(reads.values())
            if len(tuples) == 0:
                n_pos = len(locus.positions)
                chars = np.empty((0, n_pos), dtype="U1")
                quals = np.empty((0, n_pos), dtype=np.int16)
            else:
                chars = np.array([tup[0] for tup in tuples])
                quals = np.array([tup[1] for tup in tuples])
            data[id] = (chars, quals)

    return data


def encode_read_alleles(locus, chars):
    """Encode read characters as integer calls based on a locus.

    Parameters
    ----------
    locus : Locus
        A locus object defining a genomic locus with known variants.
    chars : ndarray, str
        An array of base call strings.

    Returns
    -------
    calls : ndarray, int
        Characters called as reference and alternate alleles.

    """
    return _as_allelic(chars, alleles=locus.alleles)


def encode_read_distributions(locus, calls, quals=None, error_rate=0.0):
    """Encode allele calls as allele probabilities based on base
    qual scores and an additional error rate.

    Parameters
    ----------
    locus : Locus
        A locus object defining a genomic locus with known variants.
    calls : ndarray, int
        Integer encoded allele calls.
    quals : ndarray, int
        Integer call qual scores.
    error_rate : float
        Additional error to add to qual based error rate.

    Returns
    -------
    dists : ndarray, float
        Probabilities for each allele per variant in the locus.

    """
    # handle case of zero reads
    n_reads, n_pos = calls.shape
    n_alleles = locus.count_alleles()
    if n_reads == 0:
        max_allele = int(np.max(locus.count_alleles(), initial=0))
        encoded = np.empty((n_reads, n_pos, max_allele), dtype=float)
        return encoded

    # convert error_rate to prob of correct call
    probs = np.ones((calls.shape), dtype=float) * (1 - error_rate)

    # convert qual scores to probs and multiply
    if quals is not None:
        assert calls.shape == quals.shape
        probs *= util.prob_of_qual(quals)

    encoded = _as_probabilistic(calls, n_alleles, probs)
    return encoded
