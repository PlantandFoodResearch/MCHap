import sys
import argparse
import pysam
import warnings
import numpy as np
import pandas as pd
from numba import njit, vectorize

from mchap.application import arguments
from mchap.io.vcf import headermeta
from mchap.io.vcf import infofields
from mchap.io.vcf.util import vcfstr


@vectorize(nopython=True)
def _ord_to_index(a):
    if (a == 65) or (a == 97):
        # A
        i = 0
    elif (a == 67) or (a == 99):
        # C
        i = 1
    elif (a == 71) or (a == 103):
        # G
        i = 2
    elif (a == 84) or (a == 116):
        # T
        i = 3
    else:
        i = -1
    return i


def bases_to_indices(alleles):
    alleles = np.array(alleles, copy=False, dtype="|S1")
    alleles.dtype = np.int8
    return _ord_to_index(alleles)


@njit
def _count_alleles(zeros, alleles):
    n = len(alleles)
    for i in range(n):
        a = alleles[i]
        zeros[a] += 1
    return


def bam_samples(alignment_files, tag="SM"):
    out = [None] * len(alignment_files)
    for i, bam in enumerate(alignment_files):
        read_groups = bam.header["RG"]
        sample_id = read_groups[0][tag]
        if len(read_groups) > 1:
            for rg in read_groups:
                if rg[tag] != sample_id:
                    raise ValueError(
                        "Expected one sample per bam but found {} and {} in {}".format(
                            sample_id, rg[tag], bam.filename.decode()
                        )
                    )
        out[i] = sample_id
    return np.array(out)


def bam_region_depths(
    alignment_files,
    contig,
    start,
    stop,
    dtype=np.int64,
    **kwargs,
):
    n_samples = len(alignment_files)
    n_pos = stop - start
    shape = (n_pos, n_samples, 4)
    depths = np.zeros(shape, dtype=dtype)
    for j, bam in enumerate(alignment_files):
        for column in bam.pileup(
            contig=contig,
            start=start,
            stop=stop,
            truncate=True,
            multiple_iterators=False,
            **kwargs,
        ):
            # if start <= column.pos < stop:
            i = column.pos - start
            alleles = column.get_query_sequences()
            if isinstance(alleles, list):
                alleles = bases_to_indices(alleles)
                _count_alleles(depths[i, j], alleles)
    return depths


def write_vcf_header(command, reference):
    format_header = str(headermeta.fileformat("v4.3"))
    date_header = str(headermeta.filedate())
    source_header = str(headermeta.source())
    command_header = str(headermeta.commandline(command))
    reference_header = str(headermeta.reference(reference.filename.decode()))
    info_header = "\n".join(
        [str(infofields.REFMASKED), str(infofields.AD), str(infofields.ADMF)]
    )
    contig_header = "\n".join(
        str(headermeta.ContigHeader(s, i))
        for s, i in zip(reference.references, reference.lengths)
    )
    columns_header = "#" + "\t".join(
        ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    )
    string = (
        "\n".join(
            [
                format_header,
                date_header,
                source_header,
                command_header,
                reference_header,
                contig_header,
                info_header,
                columns_header,
            ]
        )
        + "\n"
    )
    sys.stdout.write(string)


def sort_chunk_alleles(allele_depth, reference_index):
    """Sort allele depths so that the reference allele is first
    followed by alternate alleles in decending order.
    """
    n_variants, _, n_alleles = allele_depth.shape
    assert reference_index.shape == (n_variants,)
    total_depth = allele_depth.sum(axis=1)
    order = np.argsort(total_depth, axis=-1)[:, ::-1].astype(int)
    reference_index = reference_index[:, None]
    not_ref = order != reference_index
    alt_order = order.ravel()[not_ref.ravel()].reshape(n_variants, n_alleles - 1)
    order = np.hstack([reference_index, alt_order])
    sorted_depth = np.zeros_like(allele_depth)
    # TODO: improve efficiency
    for i in range(n_variants):
        sorted_depth[i] = allele_depth[i][:, order[i]]
    # alleles
    chars = np.array(["A", "C", "G", "T"])
    sorted_chars = chars[order]
    return sorted_chars, sorted_depth


def write_vcf_block(
    contig,
    start,
    stop,
    reference,
    alignment_files,
    maf,
    mapping_quality,
    skip_duplicates,
    skip_qcfail,
    skip_supplementary,
):
    # process reference info
    assert start < stop
    variant_position = np.arange(start, stop)
    variant_contig = np.full(len(variant_position), contig)
    variant_reference = np.array(list(reference.fetch(contig, start, stop).upper()))
    variant_reference_index = bases_to_indices(variant_reference)
    unsorted_allele_depth = bam_region_depths(
        alignment_files,
        contig,
        start,
        stop,
        dtype=np.int64,
        min_quality=mapping_quality,
        skip_duplicates=skip_duplicates,
        skip_qcfail=skip_qcfail,
        skip_supplementary=skip_supplementary,
    )
    # TODO: better work around for unknown reference allele
    idx = variant_reference_index >= 0
    if np.any(~idx):
        variant_position = variant_position[idx]
        variant_contig = variant_contig[idx]
        variant_reference = variant_reference[idx]
        variant_reference_index = variant_reference_index[idx]
        unsorted_allele_depth = unsorted_allele_depth[idx]
    if len(variant_position) < 1:
        # no usable variants
        return
    # sorting
    allele_char, allele_depth = sort_chunk_alleles(
        unsorted_allele_depth, variant_reference_index
    )
    # reference allele should be first
    np.all(variant_reference == allele_char[:, 0])
    # frequencies
    with np.errstate(divide="ignore", invalid="ignore"):
        allele_freq = allele_depth / allele_depth.sum(axis=-1, keepdims=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        pop_freq = np.nanmean(allele_freq, axis=1)
    # filter by MAF
    keep = pop_freq >= maf
    # remove any monomrphic
    idx = keep.sum(axis=-1) > 1
    if idx.sum() == 0:
        # no segregating variants
        return
    variant_contig = variant_contig[idx]
    variant_position = variant_position[idx]
    allele_char = allele_char[idx]
    allele_depth = allele_depth[idx]
    pop_freq = pop_freq[idx]
    keep = keep[idx]
    # mask ref if not identified
    reference_masked = ~keep[:, 0]
    keep[:, 0] = True  # reference allele
    # filter minor alleles
    allele_char[~keep] = ""
    # append
    n = len(variant_contig)
    null = np.full(n, ".")
    pop_depth = allele_depth.sum(axis=1)
    reference_allele = allele_char[:, 0]
    alternate_alleles = allele_char[:, 1:]
    # TODO: improve efficiency
    alternate_alleles = np.array(
        [",".join([c for c in row if c]) for row in alternate_alleles]
    )
    # info fileds, TODO: improve efficiency
    info = [
        "AD=" + vcfstr(d[k]) + ";ADMF=" + vcfstr(f[k])
        for d, f, k in zip(pop_depth, pop_freq.round(3), keep)
    ]
    for i, b in enumerate(reference_masked):
        if b:
            info[i] = "REFMASKED;" + info[i]
    table = pd.DataFrame(
        {
            "CHROM": variant_contig,
            "POS": variant_position + 1,  # VCF is 1-based
            "ID": null,
            "REF": reference_allele,
            "ALT": alternate_alleles,
            "QUAL": null,
            "FILTER": null,
            "INFO": info,
            # "FORMAT": null,
        }
    )
    table.to_csv(sys.stdout, sep="\t", index=False, header=False)


def main(command):
    # parse arguments
    parser = argparse.ArgumentParser("WARNING this tool is experimental")
    args = [
        arguments.basis_targets,
        arguments.reference,
        arguments.bam,
        arguments.find_snvs_maf,
        arguments.read_group_field,
        arguments.mapping_quality,
        arguments.skip_duplicates,
        arguments.skip_qcfail,
        arguments.skip_supplementary,
    ]
    for arg in args:
        arg.add_to(parser)
    if len(command) < 3:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(command[2:])
    bed_path = args.targets[0]
    bed = pd.read_table(bed_path, header=None)[[0, 1, 2]]
    bed.columns = ["contig", "start", "stop"]
    reference_path = args.reference[0]
    reference = pysam.Fastafile(reference_path)
    samples, sample_bams = arguments.parse_sample_bam_paths(
        args.bam, None, args.read_group_field[0]
    )
    # validate bam files
    samples = np.array(samples)
    bam_paths = np.array(
        [sample_bams[s][0][1] for s in samples]
    )  # TODO: this is horrible!
    # create alignment file objects and reuse them throughout
    # this is important for cram performance!
    # also pass reference name explicitly for robustness
    alignment_files = [
        pysam.AlignmentFile(path, reference_filename=reference_path)
        for path in bam_paths
    ]
    samples_found = bam_samples(alignment_files, tag=args.read_group_field[0]).astype(
        "U"
    )
    mismatch = samples_found != samples
    if np.any(mismatch):
        raise IOError(
            "Samples ({}) did not match bam files ({})".format(
                samples[mismatch], bam_paths[mismatch]
            )
        )
    # generate and write header
    write_vcf_header(command, reference)
    for _, interval in bed.iterrows():
        write_vcf_block(
            interval.contig,
            interval.start,
            interval.stop,
            reference,
            alignment_files,
            maf=args.maf[0],
            mapping_quality=args.mapping_quality[0],
            skip_duplicates=args.skip_duplicates,
            skip_qcfail=args.skip_qcfail,
            skip_supplementary=args.skip_supplementary,
        )
