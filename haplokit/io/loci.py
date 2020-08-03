#!/usr/bin/env python3

import numpy as np
import gzip
import pysam
from dataclasses import dataclass

from haplokit.encoding import allelic


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
                if chars[idx+offset] != char:
                    message = 'Reference allele does not match sequence at position {}:{}'
                    raise ValueError(message.format(self.contig, pos + offset))
                
                # remove chars
                chars[idx+offset] = ''
                
            # add template position
            chars[idx] = '{}'
        
        # join and return
        return ''.join(chars)

    def format_haplotypes(self, array, gap='-'):
        """Format integer encoded alleles as a haplotype string"""
        variants = allelic.as_characters(array, gap=gap, alleles=self.alleles)
        template = self._template_sequence()
        return [template.format(*hap) for hap in variants]

    def format_variants(self, array, gap='-'):
        """Format integer encoded alleles as a haplotype string"""
        return allelic.as_characters(array, gap=gap, alleles=self.alleles)


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
        variants=None
    )


def read_bed4(bed, region=None):

    if region:
        # must be gzipped and tabix indexed
        if isinstance(region, str):
            region = (region, )
        with pysam.TabixFile(bed) as tbx:
            for line in tbx.fetch(*region):
                yield _parse_bed4_line(line)

    else:
        # check if gzipped
        with open(bed, 'rb') as f:
            token = f.read(3)
            f.seek(0)
            if token == b'\x1f\x8b\x08':
                # is gzipped
                f =  gzip.GzipFile(fileobj=f)
            else:
                # not gzipped so assume plain text
                pass
            for line in f:
                if line.startswith(b'#'):
                    pass
                else:
                    yield _parse_bed4_line(line.decode())


def _set_locus_sequence(locus, fasta_file):
    sequence = fasta_file.fetch(locus.contig, locus.start, locus.stop).upper()
    locus = locus.set(sequence=sequence)
    return locus


def _set_locus_variants(locus, variant_file):
    variants = []

    for var in variant_file.fetch(locus.contig, locus.start, locus.stop):
        if var.stop - var.start == 1:
            # SNP
            variants.append(
                SNP(
                    contig=var.contig,
                    start=var.start,
                    stop=var.stop,
                    name=var.id if var.id else '.',
                    alleles=(var.ref, ) + var.alts,
                )
            )
        else:
            # not a SNP
            pass

    variants=tuple(variants)
    locus = locus.set(variants=variants)
    return locus


def read_loci(bed, vcf, fasta, region=None, drop_non_variable=True):
    vcf_file = pysam.VariantFile(vcf)
    ref_file = pysam.FastaFile(fasta)

    for locus in read_bed4(bed, region=region):
        locus = _set_locus_variants(locus, vcf_file)

        if drop_non_variable and len(locus.variants) == 0:
            pass

        else:
            yield _set_locus_sequence(locus, ref_file)

    vcf_file.close()
    ref_file.close()
