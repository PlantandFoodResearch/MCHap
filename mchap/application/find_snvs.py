# NOTE: This file contains commented out functions and for genotype calling.
# This functionality may be added in a future release after sufficient testing.

import sys
import argparse
import pysam
import numpy as np
import pandas as pd

# from math import lgamma
from numba import njit, vectorize, guvectorize

from mchap.application import arguments
from mchap.io.vcf import headermeta
from mchap.io.vcf import infofields
from mchap.io.vcf import formatfields
from mchap.io.vcf.util import vcfstr

# from mchap.jitutils import (
#     add_log_prob,
#     comb_with_replacement,
#     index_as_genotype_alleles,
# )


# @njit(cache=True)
# def _increment_allele_counts(allele_count):
#     n_allele = len(allele_count)
#     for i in range(n_allele):
#         if i == n_allele:
#             # final genotype
#             raise (ValueError, "Final genotype")
#         ci = allele_count[i]
#         if ci == 0:
#             pass
#         else:
#             allele_count[i] = 0
#             allele_count[i + 1] += 1
#             allele_count[0] = ci - 1
#             return


# @njit(cache=True)
# def _genotype_log_likelihood(
#     allele_depth,
#     allele_count,
#     error_rate,
#     ploidy,
#     n_allele,
# ):
#     llk = 0.0
#     for i in range(n_allele):
#         depth_i = allele_depth[i]
#         if depth_i > 0:
#             # probability of drawing allele i from genotype
#             allele_prob = 0.0
#             for j in range(n_allele):
#                 if i == j:
#                     prob = 1 - error_rate
#                 else:
#                     prob = error_rate / 4
#                 allele_prob += prob * allele_count[j] / ploidy
#             llk += np.log(allele_prob) * depth_i
#     return llk


# @njit(cache=True)
# def _log_allele_count_multinomial_prior(allele_count, frequencies, n_allele, ploidy):
#     log_num = lgamma(ploidy + 1)
#     log_denom = 0.0
#     for i in range(n_allele):
#         count = allele_count[i]
#         if count > 0:
#             log_freq = np.log(frequencies[i])
#             log_num += log_freq * count
#             log_denom += lgamma(count + 1)
#     return log_num - log_denom


# @njit(cache=True)
# def _log_allele_count_dirmul_prior(allele_count, alphas, n_allele, ploidy):
#     # Dirichlet-multinomial
#     # left side of equation in log space
#     sum_alphas = alphas.sum()
#     num = lgamma(ploidy + 1) + lgamma(sum_alphas)
#     denom = lgamma(ploidy + sum_alphas)
#     left = num - denom

#     # right side of equation
#     prod = 0.0  # log(1.0)
#     for i in range(n_allele):
#         count = allele_count[i]
#         if count > 0:
#             alpha = alphas[i]
#             num = lgamma(count + alpha)
#             denom = lgamma(count + 1) + lgamma(alpha)
#             prod += num - denom

#     # return as log probability
#     return left + prod


# @njit(cache=True)
# def _call_genotype(
#     allele_depth,
#     ploidy,
#     inbreeding,
#     frequencies,
#     n_allele,
#     error_rate,
# ):
#     allele_count = np.zeros(len(frequencies), np.int64)
#     allele_count[0] = ploidy  # first genotype
#     n_genotype = comb_with_replacement(n_allele, ploidy)
#     if inbreeding != 0.0:
#         alphas = frequencies * ((1 - inbreeding) / inbreeding)
#         use_drimul = True
#     else:
#         alphas = frequencies  # avoid zero division and typ errors
#         use_drimul = False

#     # initial values
#     current_mode = -1
#     current_mode_probability = -np.inf
#     denominator = -np.inf  # normalizing constant
#     i_final = n_genotype - 1
#     for i in range(n_genotype):
#         llk = _genotype_log_likelihood(
#             allele_depth, allele_count, error_rate, ploidy, n_allele
#         )
#         if use_drimul:
#             lprior = _log_allele_count_dirmul_prior(
#                 allele_count, alphas, n_allele, ploidy
#             )
#         else:
#             lprior = _log_allele_count_multinomial_prior(
#                 allele_count, frequencies, n_allele, ploidy
#             )
#         lprob = llk + lprior
#         if lprob > current_mode_probability:
#             current_mode = i
#             current_mode_probability = lprob
#         denominator = add_log_prob(denominator, lprob)
#         if i < i_final:
#             _increment_allele_counts(allele_count)
#     mode_genotype = index_as_genotype_alleles(current_mode, ploidy)
#     mode_probability = np.exp(current_mode_probability - denominator)
#     return mode_genotype, mode_probability


# # avoid caching of guvectorized functions as this seems to result in core dumps when in parallel on HPC
# @guvectorize(
#     [
#         "void(int64[:], int64, float64, float64[:], int64, float64, int64[:], int64[:], float64[:])",
#     ],
#     "(a),(),(),(a),(),(),(k)->(k),()",
#     nopython=True,
# )
# def call_genotype(
#     allele_depth,
#     ploidy,
#     inbreeding,
#     frequencies,
#     n_allele,
#     error_rate,
#     _,
#     mode_genotype,
#     mode_probability,
# ):
#     genotype, probability = _call_genotype(
#         allele_depth,
#         ploidy,
#         inbreeding,
#         frequencies,
#         n_allele,
#         error_rate,
#     )
#     mode_genotype[0:ploidy] = genotype
#     mode_genotype[ploidy:] = -2
#     mode_probability[0] = probability


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


@njit(cache=True)
def _count_alleles(zeros, alleles):
    n = len(alleles)
    for i in range(n):
        a = alleles[i]
        if a >= 0:
            zeros[a] += 1
    return


def bam_samples(bam_paths, reference_path, tag="SM"):
    out = [None] * len(bam_paths)
    for i, path in enumerate(bam_paths):
        with pysam.AlignmentFile(path, reference_filename=reference_path) as bam:
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
    bam_paths,
    reference_path,
    contig,
    start,
    stop,
    dtype=np.int64,
    **kwargs,
):
    n_samples = len(bam_paths)
    n_pos = stop - start
    shape = (n_pos, n_samples, 4)
    depths = np.zeros(shape, dtype=dtype)
    for j, path in enumerate(bam_paths):
        with pysam.AlignmentFile(path, reference_filename=reference_path) as bam:
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


def write_vcf_header(
    command, reference_path, info_fields=None, format_fields=None, samples=None
):
    vcfversion_header = str(headermeta.fileformat("v4.3"))
    date_header = str(headermeta.filedate())
    source_header = str(headermeta.source())
    command_header = str(headermeta.commandline(command))
    with pysam.FastaFile(reference_path) as reference:
        reference_header = str(headermeta.reference(reference.filename.decode()))
        contig_header = "\n".join(
            str(headermeta.ContigHeader(s, i))
            for s, i in zip(reference.references, reference.lengths)
        )
    components = [
        vcfversion_header,
        date_header,
        source_header,
        command_header,
        reference_header,
        contig_header,
    ]
    if info_fields is not None:
        info_header = "\n".join([str(f) for f in info_fields])
        components += [info_header]
    if format_fields is not None:
        format_header = "\n".join([str(f) for f in format_fields])
        components += [format_header]

    columns_header = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
    if samples is not None:
        columns_header += ["FORMAT"] + list(samples)
    columns_header = "#" + "\t".join(columns_header)
    components += [columns_header]
    string = "\n".join(components) + "\n"
    sys.stdout.write(string)


# avoid caching of guvectorized functions as this seems to result in core dumps when in parallel on HPC
@guvectorize(
    [
        "void(boolean[:], int64[:], boolean[:])",
        "void(int64[:], int64[:], int64[:])",
        "void(float64[:], int64[:], float64[:])",
    ],
    "(n),(n)->(n)",
    nopython=True,
)
def _order_by(values, order, out):
    out[:] = values[order]


def _vcf_sort_alleles(frequencies, reference_index):
    n_variants, n_alleles = frequencies.shape
    order = np.argsort(frequencies, axis=-1, kind="stable")[:, ::-1].astype(int)
    reference_index = reference_index[:, None]
    not_ref = order != reference_index
    alt_order = order.ravel()[not_ref.ravel()].reshape(n_variants, n_alleles - 1)
    order = np.hstack([reference_index, alt_order])
    return order


def _order_as_vcf_alleles(order, keep):
    chars = np.array(["A", "C", "G", "T"], dtype="|S1")
    chars = chars[order]
    chars = np.where(keep, chars, b"")
    ref = chars[:, 0].astype("U")
    alts = chars[:, 1:]
    n = alts.shape[-1]
    alts.dtype = np.dtype(f"|S{n}")
    alts = np.char.join(",", alts.ravel().astype("U"))
    return ref, alts


def format_allele_counts(counts, keep, sep=","):
    n_variant, n_sample, n_allele = counts.shape
    if keep.ndim == 2:
        keep = keep[:, None, :]
    keep = np.broadcast_to(keep, (n_variant, n_sample, n_allele))
    chars = counts.astype("U")
    chars = np.where(keep, chars, "")
    out = chars[:, :, 0]
    seps = np.where(keep, sep, "")
    for i in range(1, n_allele):
        out = np.char.add(out, seps[:, :, i])
        out = np.char.add(out, chars[:, :, i])
    return out


def format_genotype_calls(calls, sep="/"):
    _, _, max_ploidy = calls.shape
    chars = calls.astype("U")
    unknown = calls == -1
    chars = np.where(unknown, ".", chars)
    pad = calls <= -2
    chars = np.where(pad, "", chars)
    out = chars[:, :, 0]
    seps = np.where(pad, "", sep)
    for i in range(1, max_ploidy):
        out = np.char.add(out, seps[:, :, i])
        out = np.char.add(out, chars[:, :, i])
    return out


def format_floats(floats, precision=3):
    string = floats.round(precision).astype("U")
    # remove any ".0" from strings
    string = np.char.rstrip(string, "0")
    string = np.char.rstrip(string, ".")
    string[np.isnan(floats)] = "."
    return string


def format_samples_columns(
    genotype_calls=None, genotype_probs=None, allele_depths=None, allele_keep=None
):
    # TODO: minimize unicode dtype sizes
    fields = "GT"
    if genotype_calls is None:
        strings = np.array(["."])
    else:
        strings = format_genotype_calls(genotype_calls)
    if genotype_probs is not None:
        fields += ":GPM"
        strings = np.char.add(strings, ":")
        strings = np.char.add(strings, format_floats(genotype_probs))
    if allele_depths is not None:
        fields += ":AD"
        assert allele_keep is not None
        strings = np.char.add(strings, ":")
        strings = np.char.add(strings, format_allele_counts(allele_depths, allele_keep))
    cols = pd.DataFrame(strings)
    fields = pd.DataFrame(np.full(len(strings), fields))
    return pd.concat([fields, cols], axis=1)


def write_vcf_block(
    contig,
    start,
    stop,
    reference_path,
    bam_paths,
    # sample_ploidy,
    # sample_inbreeding,
    # base_error_rate,
    # allele_frequency_prior,
    maf,
    mad,
    ind_maf,
    ind_mad,
    min_ind,
    mapping_quality,
    skip_duplicates,
    skip_qcfail,
    skip_supplementary,
):
    # process reference info
    assert start < stop
    variant_position = np.arange(start, stop)
    variant_contig = np.full(len(variant_position), contig)
    with pysam.FastaFile(reference_path) as reference:
        variant_reference = np.array(list(reference.fetch(contig, start, stop).upper()))
    variant_reference_index = bases_to_indices(variant_reference)
    allele_depth = bam_region_depths(
        bam_paths,
        reference_path,
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
        allele_depth = allele_depth[idx]
    if len(variant_position) < 1:
        # no usable variants
        return

    # frequencies
    with np.errstate(divide="ignore", invalid="ignore"):
        allele_freq = allele_depth / allele_depth.sum(axis=-1, keepdims=True)
    # filter by ind-maf and ind-mad for each sample
    keep = ((allele_freq >= ind_maf) & (allele_depth >= ind_mad)).sum(axis=1) >= min_ind
    # filter by maf
    if maf > 0.0:
        keep &= np.mean(allele_freq, axis=1) >= maf
    # filter by mad
    if mad > 0:
        keep &= np.sum(allele_depth, axis=1) >= mad
    # remove any monomorphic/missing
    idx = keep.sum(axis=-1) > 1
    if idx.sum() == 0:
        # no segregating variants
        return
    variant_contig = variant_contig[idx]
    variant_position = variant_position[idx]
    variant_reference = variant_reference[idx]
    variant_reference_index = variant_reference_index[idx]
    allele_depth = allele_depth[idx]
    allele_freq = allele_freq[idx]
    keep = keep[idx]

    # zero out non-kept alleles
    allele_freq = np.where(keep[:, None, :], allele_freq, 0.0)
    depth_mean_freq = np.nanmean(allele_freq, axis=1)

    # sort alleles by pop frequencies
    order = _vcf_sort_alleles(depth_mean_freq, variant_reference_index)
    allele_depth = _order_by(allele_depth, order[:, None, :])
    allele_freq = _order_by(allele_freq, order[:, None, :])
    depth_mean_freq = _order_by(depth_mean_freq, order)
    keep = _order_by(keep, order)

    # # calculate flat prior, this will be 0 for a masked reference allele
    # flat_prior_frequencies = keep / keep.sum(axis=-1, keepdims=True)

    # mask ref if not identified
    reference_masked = ~keep[:, 0]
    keep[:, 0] = True  # reference allele

    # create allele columns
    reference_allele, alternate_alleles = _order_as_vcf_alleles(order, keep)
    assert np.all(reference_allele == variant_reference)

    # append
    n = len(variant_contig)
    null = np.full(n, ".")
    pop_depth = allele_depth.sum(axis=1)

    # info fields, TODO: improve efficiency
    info = [
        "AD=" + vcfstr(d[k]) + ";ADMF=" + vcfstr(f[k])
        for d, f, k in zip(pop_depth, depth_mean_freq.round(3), keep)
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
        }
    )

    # # call genotypes
    # variant_unique_alleles = keep.sum(axis=-1)
    # if allele_frequency_prior == "FLAT":
    #     frequencies = flat_prior_frequencies
    # elif allele_frequency_prior == "ADMF":
    #     frequencies = depth_mean_freq
    # else:
    #     raise ValueError("--allele-frequency-prior must be one of {'FLAT', 'ADMF'}")
    # _ = np.empty(sample_ploidy.max(), int)
    # # divide by zero warning expected with any freq of zero
    # with np.errstate(divide="ignore", invalid="ignore"):
    #     calls, probs = call_genotype(
    #         allele_depth,
    #         sample_ploidy,
    #         sample_inbreeding,
    #         frequencies[:, None, :],
    #         variant_unique_alleles[:, None],
    #         base_error_rate,
    #         _,
    #     )

    # # Add sample data
    # sample_cols = format_samples_columns(calls, probs, allele_depth, keep)
    # table = pd.concat([table, sample_cols], axis=1)

    # Add sample data
    sample_cols = format_samples_columns(
        genotype_calls=None,
        genotype_probs=None,
        allele_depths=allele_depth,
        allele_keep=keep,
    )
    table = pd.concat([table, sample_cols], axis=1)

    # write block
    table.to_csv(sys.stdout, sep="\t", index=False, header=False)


def main(command):
    # parse arguments
    parser = argparse.ArgumentParser("WARNING this tool is experimental")
    args = [
        arguments.basis_targets,
        arguments.reference,
        arguments.bam,
        # arguments.ploidy,
        # arguments.inbreeding,
        # arguments.find_snvs_allele_frequency_prior,
        arguments.find_snvs_maf,
        arguments.find_snvs_mad,
        arguments.find_snvs_ind_maf,
        arguments.find_snvs_ind_mad,
        arguments.find_snvs_min_ind,
        # arguments.base_error_rate,
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
    samples, sample_bams = arguments.parse_sample_bam_paths(
        args.bam, None, args.read_group_field[0], reference_path=reference_path
    )
    # sample_ploidy = arguments.parse_sample_value_map(
    #     args.ploidy[0],
    #     samples,
    #     type=int,
    # )
    # sample_ploidy = np.array([sample_ploidy[s] for s in samples])
    # sample_inbreeding = arguments.parse_sample_value_map(
    #     args.inbreeding[0],
    #     samples,
    #     type=float,
    # )
    # sample_inbreeding = np.array([sample_inbreeding[s] for s in samples])
    # validate bam files
    samples = np.array(samples)
    bam_paths = np.array(
        [sample_bams[s][0][1] for s in samples]
    )  # TODO: this is horrible!
    # create alignment file objects and reuse them throughout
    # this is important for cram performance!
    # also pass reference name explicitly for robustness
    samples_found = bam_samples(
        bam_paths, reference_path, tag=args.read_group_field[0]
    ).astype("U")
    mismatch = samples_found != samples
    if np.any(mismatch):
        raise IOError(
            "Samples ({}) did not match bam files ({})".format(
                samples[mismatch], bam_paths[mismatch]
            )
        )
    # generate and write header
    info_fields = [infofields.REFMASKED, infofields.AD, infofields.ADMF]
    format_fields = [formatfields.GT, formatfields.AD]
    write_vcf_header(
        command,
        reference_path,
        samples=samples,
        info_fields=info_fields,
        format_fields=format_fields,
    )
    # generate and write record blocks
    for _, interval in bed.iterrows():
        write_vcf_block(
            interval.contig,
            interval.start,
            interval.stop,
            reference_path,
            bam_paths,
            # sample_ploidy,
            # sample_inbreeding,
            # base_error_rate=args.base_error_rate[0],
            # allele_frequency_prior=args.allele_frequency_prior[0],
            maf=args.maf[0],
            mad=args.mad[0],
            ind_maf=args.ind_maf[0],
            ind_mad=args.ind_mad[0],
            min_ind=args.min_ind[0],
            mapping_quality=args.mapping_quality[0],
            skip_duplicates=args.skip_duplicates,
            skip_qcfail=args.skip_qcfail,
            skip_supplementary=args.skip_supplementary,
        )
