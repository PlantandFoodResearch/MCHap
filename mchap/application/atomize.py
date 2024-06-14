import sys
import argparse
import pysam
import numpy as np
import pandas as pd

from mchap.io.vcf import headermeta as HEADER
from mchap.io.vcf import columns as COLUMN
from mchap.io.vcf import infofields as INFO
from mchap.io.vcf import formatfields as FORMAT
from mchap.application import arguments


def get_haplotype_snvs(vcf_record):
    snv_pos = np.array(vcf_record.info[INFO.SNVPOS.id]) - 1
    n_pos = len(snv_pos)
    n_hap = len(vcf_record.alts) + 1
    haplotype_snvs = np.zeros((n_hap, n_pos), dtype="U")
    haplotype_snvs[0] = np.array(list(vcf_record.ref))[snv_pos]
    for i, alt in enumerate(vcf_record.alts):
        alt = np.array(list(alt))
        haplotype_snvs[i + 1] = alt[snv_pos]
    return haplotype_snvs


def format_snv_alleles(haplotype_snvs):
    ref = haplotype_snvs[0]
    _, n_pos = haplotype_snvs.shape
    alts = []
    n_alts = []
    for i in range(n_pos):
        _, idx = np.unique(haplotype_snvs[:, i], return_index=True)
        idx.sort()
        assert idx[0] == 0
        idx = idx[1:]
        n_alts.append(len(idx))
        alts.append(",".join(haplotype_snvs[:, i][idx]))
    return ref, np.array(alts), np.array(n_alts)


def get_haplotype_snv_indices(haplotype_snvs):
    n_hap, n_pos = haplotype_snvs.shape
    haplotype_idxs = np.zeros((n_hap, n_pos), dtype=int)
    for i in range(n_pos):
        d = {}
        next_allel = 0
        for h in range(n_hap):
            char = haplotype_snvs[h, i]
            a = d.get(char)
            if a is None:
                a = next_allel
                d[char] = a
                next_allel += 1
            haplotype_idxs[h, i] = a
    return haplotype_idxs


def get_sample_snv_ACP(vcf_record, haplotype_idxs):
    _, n_pos = haplotype_idxs.shape
    n_samples = len(vcf_record.samples)
    out = np.zeros((n_pos, n_samples, 4))
    for i, s in enumerate(vcf_record.samples):
        counts = vcf_record.samples[s].get(FORMAT.ACP.id)
        if counts is None:
            freqs = vcf_record.samples[s].get(FORMAT.AFP.id)
            if freqs is None:
                out[:, i, :] = np.nan
                continue
            else:
                ploidy = len(vcf_record.samples[s][FORMAT.AFP.id])
                counts = np.array(freqs) * ploidy
        else:
            counts = np.array(counts)
        for h, c in enumerate(counts):
            for p, a in enumerate(haplotype_idxs[h]):
                out[p, i, a] += c
    return out


def format_allele_floats(array, alts_number, length="R", precision=3):
    input_dims = array.ndim
    if input_dims == 2:
        array = array[:, None, :]
    elif input_dims == 3:
        pass
    else:
        raise ValueError("Number of dimensions not supported.")
    assert length in ("R", "A")
    formatted = []
    for limit, freqs in zip(alts_number, array):
        if length == "R":
            limit += 1
        freqs = freqs[:, 0:limit]
        freqs = freqs.round(precision)
        missing = np.isnan(freqs)
        freqs = freqs.astype("U")
        freqs = np.char.rstrip(freqs, "0")
        freqs = np.char.rstrip(freqs, ".")
        freqs[missing] = "."
        head = freqs[:, 0]
        tail = freqs[:, 1:]
        for t in tail.T:
            head = np.char.add(head, ",")
            head = np.char.add(head, t)
        formatted.append(head)
    formatted = np.array(formatted)
    if input_dims == 2:
        formatted = np.squeeze(formatted, 1)
    return formatted


def get_sample_snv_DS(sample_snv_ACP, sample_ploidy):
    dose = sample_snv_ACP.copy()
    # normalise
    denom = np.nansum(dose, axis=-1, keepdims=True)
    denom = np.where(denom == 0.0, np.nan, denom)
    dose /= denom
    dose *= sample_ploidy[None, :, None]
    # remove reference allele
    dose = dose[:, :, 1:]
    return dose


def get_sample_snv_GT(vcf_record, haplotype_idxs, sep="|"):
    n_haps, n_pos = haplotype_idxs.shape
    haplotype_counts = np.zeros(n_haps)
    sample_ploidy = []
    out = []
    for s in vcf_record.samples:
        haplotype_gt = vcf_record.samples[s][FORMAT.GT.id]
        ploidy = len(haplotype_gt)
        sample_ploidy.append(ploidy)
        snv_gts = np.full((ploidy, n_pos), -1, int)
        for i, a in enumerate(haplotype_gt):
            if a is not None:
                haplotype_counts[a] += 1
                snv_gts[i] = haplotype_idxs[a]
        snv_gts = snv_gts.T
        out.append(
            [sep.join([str(a) if a >= 0 else "." for a in call]) for call in snv_gts]
        )
    out = np.array(out)
    snv_counts = np.zeros((n_pos, haplotype_idxs.max() + 1))
    for hap, c in enumerate(haplotype_counts):
        for p, a in enumerate(haplotype_idxs[hap]):
            snv_counts[p, a] += c
    return snv_counts, np.array(sample_ploidy), out.T  # variants * samples


def get_sample_snv_PQ(vcf_record):
    n_pos = len(vcf_record.info[INFO.SNVPOS.id])
    pq = np.array([d[FORMAT.SQ.id] for d in vcf_record.samples.values()]).astype("U")
    return np.tile(pq, (n_pos, 1))


def get_sample_snv_depth(vcf_record):
    p = len(vcf_record.info[INFO.SNVPOS.id])
    null = np.full(p, np.nan)
    out = []
    for s in vcf_record.samples:
        dp = vcf_record.samples[s].get(FORMAT.SNVDP.id, null)
        out.append(list(dp))
    return np.array(out).T


def format_vcf_snv_block(vcf_record):
    # check if there are any SNVs in this block
    if vcf_record.info[INFO.SNVPOS.id] == (None,):
        return None

    # allele basics
    haplotype_snvs = get_haplotype_snvs(vcf_record)
    haplotype_idxs = get_haplotype_snv_indices(haplotype_snvs)
    _, n_pos = haplotype_snvs.shape

    # column data
    ref_column, alts_column, alts_number = format_snv_alleles(haplotype_snvs)
    pos_column = np.array(vcf_record.info[INFO.SNVPOS.id]) - 1 + vcf_record.pos
    contig_column = np.repeat(vcf_record.contig, n_pos)
    rec_id = vcf_record.id
    if rec_id:
        id_column = [rec_id + "_SNV{}".format(i + 1) for i in range(n_pos)]
    else:
        id_column = "."
    block_data = pd.DataFrame()
    block_data[COLUMN.CHROM] = contig_column
    block_data[COLUMN.POS] = pos_column
    block_data[COLUMN.ID] = id_column
    block_data[COLUMN.REF] = ref_column
    block_data[COLUMN.ALT] = alts_column
    block_data[COLUMN.QUAL] = "."
    block_data[COLUMN.FILTER] = "."

    # sample data
    info_snv_count, sample_ploidy, format_GT = get_sample_snv_GT(
        vcf_record, haplotype_idxs
    )
    sample_snv_ACP = get_sample_snv_ACP(vcf_record, haplotype_idxs)
    sample_snv_dose = get_sample_snv_DS(sample_snv_ACP, sample_ploidy)
    format_DS = format_allele_floats(sample_snv_dose, alts_number, length="A")
    format_PQ = get_sample_snv_PQ(vcf_record)
    format_GQ = np.full_like(format_PQ, ".")
    sample_depth = get_sample_snv_depth(vcf_record)
    format_DP = sample_depth.astype("U")
    format_DP[format_DP == "nan"] = "."
    sample_data = format_GT
    for field in [format_GQ, format_PQ, format_DP, format_DS]:
        sample_data = np.char.add(sample_data, ":")
        sample_data = np.char.add(sample_data, field)
    sample_data = pd.DataFrame(sample_data)
    sample_data.columns = list(vcf_record.samples)

    # info data
    info_DP = sample_depth.sum(axis=1).astype("U")
    info_DP[info_DP == "nan"] = "."
    info_DP = ["{}={}".format(INFO.DP.id, counts) for counts in info_DP]
    info_AC = format_allele_floats(info_snv_count[:, 1:], alts_number, length="A")
    info_AC = ["{}={}".format(INFO.AC.id, counts) for counts in info_AC]
    population_snv_ACP = sample_snv_ACP.sum(axis=1)
    info_ACP = format_allele_floats(population_snv_ACP, alts_number, length="R")
    info_ACP = ["{}={}".format(INFO.ACP.id, counts) for counts in info_ACP]
    info_PS = np.tile("PS={}".format(vcf_record.pos), n_pos)
    info_column = [";".join(tup) for tup in zip(info_AC, info_ACP, info_DP, info_PS)]
    block_data[COLUMN.INFO] = info_column

    # add sample data last
    format_column = np.tile(
        ":".join(
            [FORMAT.GT.id, FORMAT.GQ.id, FORMAT.PQ.id, FORMAT.DP.id, FORMAT.DS.id]
        ),
        n_pos,
    )
    block_data[COLUMN.FORMAT] = format_column

    # merge
    block_data = pd.concat([block_data, sample_data], axis=1)
    return block_data


def atomize_vcf(path, command=None):
    if command is None:
        command = "atomize {}".format(path)
    vcf = pysam.VariantFile(path)

    # header metadata
    sys.stdout.write(str(HEADER.fileformat("v4.3")) + "\n")
    sys.stdout.write(str(HEADER.filedate()) + "\n")
    sys.stdout.write(str(HEADER.source()) + "\n")
    sys.stdout.write(str(HEADER.commandline(command)) + "\n")

    # write contigs
    for contig in vcf.header.contigs.values():
        sys.stdout.write(str(contig.header_record))

    # write info fields
    for field in [
        INFO.AC,
        INFO.ACP,
        INFO.DP,
        INFO.PS,
    ]:
        sys.stdout.write(str(field) + "\n")

    # write format fields
    for field in [
        FORMAT.GT,
        FORMAT.GQ,
        FORMAT.PQ,
        FORMAT.DP,
        FORMAT.DS,
    ]:
        sys.stdout.write(str(field) + "\n")

    # write columns
    columns_header = COLUMN.COLUMNS.copy()
    columns_header += list(vcf.header.samples)
    columns_header = "#" + "\t".join(columns_header)
    sys.stdout.write(columns_header + "\n")

    # write blocks
    for record in vcf:
        block = format_vcf_snv_block(record)
        if block is not None:
            block.to_csv(sys.stdout, sep="\t", index=False, header=False)

    vcf.close()


def main(command):
    # TODO: add arguments to MCHap for integration
    parser = argparse.ArgumentParser("WARNING this tool is experimental")
    arguments.Parameter(
        "haplotypes",
        dict(
            type=str,
            nargs=1,
            default=[None],
            help=(
                "VCF file containing haplotype variants to be atomized. "
                "This file must contain INFO/SNVPOS. "
                "The FORMAT/DP field will be reported if FORMAT/SNVDP is present in this file. "
                "The FORMAT/DS field will be reported if FORMAT/ACP or FORMAT/AFP is present in this file."
            ),
        ),
    ).add_to(parser)
    if len(command) < 3:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args(command[2:])
    path = args.haplotypes[0]
    atomize_vcf(path, command=command)
