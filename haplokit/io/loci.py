#!/usr/bin/env python3

import numpy as np
import pysam
from dataclasses import dataclass

from haplokit.encoding import allelic


@dataclass(frozen=True, order=True)
class Variant:
    contig: str
    start: int
    stop: int
    name: str
    category: str
    alleles: tuple


@dataclass(frozen=True, order=True)
class Locus:

    contig: str
    start: int
    stop: int
    name: str
    category: str
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
            category=self.category,
            sequence=self.sequence,
            variants=self.variants,
        )

    def set(self, **kwargs):
        data = self.as_dict()
        data.update(kwargs)
        return type(self)(**data)

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


def _set_loci_sequence(loci, fasta_file):
    for locus in loci:
        sequence = fasta_file.fetch(locus.contig, locus.start, locus.stop).upper()
        locus = locus.set(sequence=sequence)
        yield locus


def _set_loci_variants(loci, variant_file):
    for locus in loci:
        variants = []

        for var in variant_file.fetch(locus.contig, locus.start, locus.stop):
            if var.stop - var.start == 1:
                # SNP
                variants.append(
                    Variant(
                        contig=var.contig,
                        start=var.start,
                        stop=var.stop,
                        name=var.id if var.id else '.',
                        category='SNP',
                        alleles=(var.ref, ) + var.alts,
                    )
                )
            else:
                # not a SNP
                pass

        variants=tuple(variants)
        locus = locus.set(variants=variants)
        yield locus


class Bed4File(pysam.Tabixfile):
    
    def fetch(self, *args, **kwargs):
        
        names = set()
        
        lines = super().fetch(*args, **kwargs)
        
        for line_number, line in enumerate(lines):
            
            if line[0] == '#':
                pass
            
            else:
                line = line.split()

                contig = line[0].strip()
                start = int(line[1].strip())
                stop = int(line[2].strip())
                if len(line) > 3:
                    # assume bed 4 so next column in name
                    name = line[3].strip()
                else:
                    name = '{}:{}-{}'.format(contig, start, stop)

                if name in names:
                    raise ValueError('Loci with duplicate name found on line {}'.format(line_number))
                else:
                    names.add(name)

                locus = Locus(
                    contig=contig,
                    start=start,
                    stop=stop,
                    name=name,
                    category='interval',
                    sequence=None,
                    variants=None
                )

                yield locus


def read_loci(bed, vcf, fasta, region=(), drop_non_variable=True):
    bed_file = Bed4File(bed)
    vcf_file = pysam.VariantFile(vcf)
    ref_file = pysam.FastaFile(fasta)

    loci = _set_loci_sequence(
        _set_loci_variants(
            bed_file.fetch(*region),
            vcf_file,
        ),
        ref_file,
    )

    if drop_non_variable:
        for locus in loci:
            if len(locus.variants) > 0:
                yield locus
    else:
        yield from loci

    bed_file.close()
    vcf_file.close()
    ref_file.close()
