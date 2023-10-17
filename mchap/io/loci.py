#!/usr/bin/env python3

import gzip
import pysam
import numpy as np
from dataclasses import dataclass

from mchap.encoding import integer, character


__all__ = [
    "SNP",
    "Locus",
    "LocusPrior",
    "read_bed4",
]


@dataclass(frozen=True, order=True)
class SNP:
    contig: str
    start: int
    stop: int
    name: str
    alleles: tuple


@dataclass(frozen=True, order=True)
class Locus:

    contig: str
    start: int
    stop: int
    name: str
    sequence: str
    variants: tuple

    @property
    def positions(self):
        return [v.start for v in self.variants]

    @property
    def alleles(self):
        return [v.alleles for v in self.variants]

    @property
    def range(self):
        return range(self.start, self.stop)

    def count_alleles(self):
        return [len(tup) for tup in self.alleles]

    def as_dict(self):
        return dict(
            contig=self.contig,
            start=self.start,
            stop=self.stop,
            name=self.name,
            sequence=self.sequence,
            variants=self.variants,
        )

    def set(self, **kwargs):
        data = self.as_dict()
        data.update(kwargs)
        return type(self)(**data)

    def set_sequence(self, fasta):
        with pysam.FastaFile(fasta) as f:
            return _set_locus_sequence(self, f)

    def set_variants(self, vcf):
        with pysam.VariantFile(vcf) as f:
            return _set_locus_variants(self, f)

    def _template_sequence(self):
        chars = list(self.sequence)
        ref_alleles = (tup[0] for tup in self.alleles)
        for pos, string in zip(self.positions, ref_alleles):
            idx = pos - self.start
            for offset, char in enumerate(string):
                if chars[idx + offset] != char:
                    message = (
                        "Reference allele does not match sequence at position {}:{}"
                    )
                    raise ValueError(message.format(self.contig, pos + offset))

                # remove chars
                chars[idx + offset] = ""

            # add template position
            chars[idx] = "{}"

        # join and return
        return "".join(chars)

    def format_haplotypes(self, array, gap="-"):
        """Format integer encoded alleles as a haplotype string"""
        variants = integer.as_characters(array, gap=gap, alleles=self.alleles)
        template = self._template_sequence()
        return [template.format(*hap) for hap in variants]

    def format_variants(self, array, gap="-"):
        """Format integer encoded alleles as a haplotype string"""
        return integer.as_characters(array, gap=gap, alleles=self.alleles)

    @classmethod
    def from_region_string(cls, string, name=None):
        contig, interval = string.strip().split(":")
        start, stop = interval.strip().split("-")
        return cls(
            contig=contig,
            start=int(start),
            stop=int(stop),
            name=name,
            sequence=None,
            variants=None,
        )


@dataclass(frozen=True, order=True)
class LocusPrior(Locus):
    alts: tuple
    frequencies: np.ndarray
    mask_reference_allele: bool = False

    def set(self):
        raise NotImplementedError

    def set_sequence(self):
        raise NotImplementedError

    def set_variants(self):
        raise NotImplementedError

    def encode_haplotypes(self):
        strings = (self.sequence,) + self.alts
        chars = np.array([list(string) for string in strings])
        idx = np.array(self.positions) - self.start
        if len(idx) == 0:
            return np.zeros((len(strings), 0), dtype=int)
        return character.as_allelic(chars[:, idx], self.alleles)

    @classmethod
    def from_variant_record(
        cls,
        record,
        use_snvpos=False,
        frequency_tag=None,
        frequency_min=None,
        masked_reference_flag="REFMASKED",
    ):
        """Generate a locusPrior object with reference and variants from
        a known MNP.

        Parameters
        ---------
        record : VariantRecord.
            A pysam VariantRecord object for an MNP variant.
        use_snvpos : bool
            If true then the "SNVPOS" info tag will be used to
            identify positions of SNV variants rather than identifying
            them directly from the ref and alt sequences.
        frequency_tag : str
            VCF INFO tag to estimate allele frequencies from.
        frequency_min : float
            Minimum frequency required to include an alternate allele.
            If the reference allele does not meet this threshold then
            it will be included with a frequency of 0.
        masked_reference_flag : str
            VCF INFO tag used to indicate that the reference allele should
            not be used.

        Returns
        -------
        locus : Locus
            A locus object with populated sequence and variants fields.
        """
        ref_length = len(record.ref)
        sequences = (record.ref,)
        if record.alts:
            assert all(ref_length == len(alt) for alt in record.alts)
            alts = record.alts
        else:
            alts = ()
        sequences += alts

        # prior allele frequencies
        if frequency_tag:
            frequencies = np.array(record.info[frequency_tag])
            assert len(frequencies) == len(sequences)
        else:
            # flat prior
            frequencies = np.ones(len(sequences)) / len(sequences)

        # check if ref allele is masked and make frequency match
        if masked_reference_flag in record.info:
            mask_reference_allele = record.info[masked_reference_flag]
            if mask_reference_allele:
                frequencies[0] = 0
        else:
            mask_reference_allele = False

        # normalise frequencies
        denom = frequencies.sum()
        if denom > 0:
            frequencies /= denom
        else:
            frequencies[:] = np.nan

        # mask rare haplotypes
        if frequency_min:
            keep = frequencies >= frequency_min
            if not keep[0]:
                # must keep ref so mask
                mask_reference_allele = True
                frequencies[0] = 0
                keep[0] = True
            # drop alts
            sequences = tuple(a for a, k in zip(sequences, keep) if k)
            frequencies = frequencies[keep]
            # re-normalise frequencies
            denom = frequencies.sum()
            if denom > 0:
                frequencies /= denom
            else:
                frequencies[:] = np.nan
        assert len(frequencies) == len(sequences)

        # encoded haplotypes
        haplotypes = np.array([list(var) for var in sequences])
        if use_snvpos:
            snvpos = record.info["SNVPOS"]
            if snvpos == (None,):
                snvpos = ()
            positions = np.array(snvpos, int) - 1  # 1-based index in VCF
        else:
            positions = np.where((haplotypes != haplotypes[0:1]).any(axis=0))[0]
        snp_alleles = haplotypes[:, positions].T
        snps = []
        for offset, alleles in zip(positions, snp_alleles):
            _, idx = np.unique(alleles, return_index=True)
            idx.sort()
            alleles = tuple(alleles[idx])
            pos = offset + record.start
            snps.append(SNP(record.chrom, pos, pos + 1, ".", alleles=alleles))
        return cls(
            contig=record.chrom,
            start=record.start,
            stop=record.stop,
            name=record.id if record.id else ".",
            sequence=record.ref,
            variants=tuple(snps),
            alts=sequences[1:],
            frequencies=frequencies,
            mask_reference_allele=mask_reference_allele,
        )


def _parse_bed4_line(line):
    line = line.split()
    contig = line[0].strip()
    start = int(line[1].strip())
    stop = int(line[2].strip())
    if len(line) > 3:
        # assume bed 4 so next column in name
        name = line[3].strip()
    else:
        name = None

    return Locus(
        contig=contig,
        start=start,
        stop=stop,
        name=name,
        sequence=None,
        variants=None,
    )


def read_bed4(bed, region=None):

    if region:
        # must be gzipped and tabix indexed
        if isinstance(region, str):
            region = (region,)
        with pysam.TabixFile(bed) as tbx:
            for line in tbx.fetch(*region):
                yield _parse_bed4_line(line)

    else:
        # check if gzipped
        with open(bed, "rb") as f:
            token = f.read(3)
            f.seek(0)
            if token == b"\x1f\x8b\x08":
                # is gzipped
                f = gzip.GzipFile(fileobj=f)
            else:
                # not gzipped so assume plain text
                pass
            for line in f:
                if line.startswith(b"#"):
                    pass
                else:
                    yield _parse_bed4_line(line.decode())


def _set_locus_sequence(locus, fasta_file):
    sequence = fasta_file.fetch(locus.contig, locus.start, locus.stop).upper()
    locus = locus.set(sequence=sequence)
    return locus


def _merge_snps(x, y):
    match = [
        x.contig == y.contig,
        x.name == y.name,
        x.start == y.start,
        x.stop == y.stop,
        x.alleles[0] == y.alleles[0],
    ]

    if not all(match):
        x_str = "{}: {}:{}".format(x.name, x.contig, x.start)
        y_str = "{}: {}:{}".format(y.name, y.contig, y.start)
        message = 'Cannot merge SNPs "{}" and "{}"'.format(x_str, y_str)
        raise ValueError(message)

    alleles = x.alleles
    alleles += tuple(a for a in y.alleles if a not in alleles)

    return SNP(
        contig=x.contig,
        start=x.start,
        stop=x.stop,
        name=x.name,
        alleles=alleles,
    )


def _set_locus_variants(locus, variant_file):
    variants = []
    positions = set()

    for var in variant_file.fetch(locus.contig, locus.start, locus.stop):
        alleles = (var.ref,) + var.alts

        if (var.stop - var.start == 1) and all(len(a) == 1 for a in alleles):
            # is a SNP
            snp = SNP(
                contig=var.contig,
                start=var.start,
                stop=var.stop,
                name=var.id if var.id else ".",
                alleles=alleles,
            )
            if snp.start in positions:
                # attempt to merge duplicates
                variants = [
                    _merge_snps(s, snp) if s.start == snp.start else s for s in variants
                ]
            else:
                variants.append(snp)
                positions.add(snp.start)
        else:
            pass

    variants = tuple(variants)
    locus = locus.set(variants=variants)
    return locus
