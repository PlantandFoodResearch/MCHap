#!/usr/bin/env python3

import pysam
from dataclasses import dataclass
from Bio import bgzf

from haplohelper.encoding import allelic


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
        

def contig_value(contig):
    chars = ''
    digits = ''
    for i in contig:
        if i.isdigit():
            digits += i
        else:
            chars += i
    return chars, int(digits)


def _template_sequence(locus):
    chars = list(locus.sequence)
    ref_alleles = (tup[0] for tup in locus.alleles)
    for pos, string in zip(locus.positions, ref_alleles):
        idx = pos - locus.start
        for offset, char in enumerate(string):
            if chars[idx+offset] != char:
                message = 'Reference allele does not match sequence at position {}:{}'
                raise ValueError(message.format(locus.contig, pos + offset))
            
            # remove chars
            chars[idx+offset] = ''
            
        # add template position
        chars[idx] = '{}'
    
    # join and return
    return ''.join(chars)


def format_haplotypes(locus, array, gap='-'):
    """Format integer encoded alleles as a haplotype string"""
    variants = allelic.as_characters(array, gap=gap, alleles=locus.alleles)
    template = _template_sequence(locus)
    return [template.format(*hap) for hap in variants]


def format_variants(locus, array, gap='-'):
    """Format integer encoded alleles as a haplotype string"""
    return allelic.as_characters(array, gap=gap, alleles=locus.alleles)


def write_loci(loci, path):
    columns = (
        'CHROM',
        'START',
        'STOP',
        'NAME',
        'CATEGORY',
        'REF',
        'ALT',
        'WITHIN',
    )

    loci_types = set()
    variant_types = set()

    # need to iter twice to get types
    for locus in loci:
        loci_types.add(locus.category)
        for var in locus.variants:
            variant_types.add(var.category)

    # make sure loci and variants are never the same type
    intercept = loci_types & variant_types
    if intercept:
        raise IOError('Found both loci and variants of types "{}"'.format(intercept))

    # use bgzip for compression
    compress = path.endswith('.gz') 
    open_ = bgzf.open if compress else open

    with open_(path, 'w') as f:

        # write header line:
        f.write('#HaploHelper Loci Format v0.01\n')
        f.write('#LOCI=' + ';'.join(loci_types) + '\n')
        f.write('#VARIANTS=' + ';'.join(variant_types) + '\n')
        f.write('#' + '\t'.join(columns) + '\n')

        for locus in loci:

            if locus.name in {None, '.'}:
                locus_name = '{}:{}-{}'.format(locus.contig, locus.start, locus.stop)
            else:
                locus_name = locus.name

            # write locus line:
            line = '\t'.join([
                locus.contig,
                str(locus.start),
                str(locus.stop),
                locus_name,
                locus.category,
                locus.sequence if locus.sequence else '.',
                '.',
                '.'
            ]) + '\n'
            f.write(line)

            for var in locus.variants:

                line = '\t'.join([
                    var.contig,
                    str(var.start),
                    str(var.stop),
                    var.name,
                    var.category,
                    var.alleles[0],
                    ','.join(var.alleles[1:]),
                    locus_name,
                ]) + '\n'
                f.write(line)


def read_loci(path, skip_non_variable=True):

    loci_types = set()
    variant_types = set()

    loci_data = {}

    # expect bgzip for compression
    compress = path.endswith('.gz') 
    open_ = bgzf.open if compress else open

    with open_(path, 'r') as f:
        for line_number, line in enumerate(f):
            
            if line.startswith('#LOCI='):
                line = line.split('=')[1]
                loci_types |= {string.strip() for string in line.split(';')}

            elif line.startswith('#VARIANTS='):
                line = line.split('=')[1]
                variant_types |= {string.strip() for string in line.split(';')}

            elif line[0] == '#':
                pass
            
            else:
                line = line.split()

                if line[4] in loci_types:
                    # this is a locus line

                    locus = {
                        'contig': line[0],
                        'start': int(line[1]),
                        'stop': int(line[2]),
                        'name': line[3],
                        'category': line[4],
                        'variants': [],
                        'sequence': line[5]
                    }

                    locus_name = locus['name']

                    if locus_name is '.':
                        raise IOError('Unnamed Locus on line {}'.format(line_number))
                    
                    elif locus_name in loci_data:
                        raise IOError('Duplicate Locus on line {}'.format(line_number))

                    loci_data[locus_name] = locus
                
                else:
                    # assume this is a variant line

                    parent = line[7]

                    if parent is '.':
                        raise IOError('Variant without locus on line {}'.format(line_number))

                    elif parent not in loci_data:
                        raise IOError('Variant on line {} found before Locus "{}"'.format(line_number, parent))

                    var = Variant(
                        contig=line[0],
                        start=int(line[1]),
                        stop=int(line[2]),
                        name=line[3],
                        category=line[4],
                        alleles=tuple((line[5] + ',' + line[6]).split(',')),
                    )

                    #locus_interval = loci[locus_name].range
                    #if pos not in locus_interval:
                    #    raise IOError('Variant on line {} not within Locus "{}"'.format(line_number, locus_name))

                    loci_data[parent]['variants'].append(var)

    loci = {}
    for name, data in loci_data.items():
        data['variants'] = tuple(data['variants'])
        locus = Locus(**data)
        r = locus.range
        for var in locus.variants:
            if (var.start not in r) or ((var.stop - 1) not in r):
                raise IOError('Variant location {}-{} not within Locus "{}"'.format(var.start, var.stop, name))
        loci[name] = locus
        if skip_non_variable and len(locus.variants) == 0:
            del loci[name]

    return list(loci.values())


def read_bed(path, locus_category='interval'):

    names = set()

    with open(path, 'r') as f:
        for line_number, line in enumerate(f): 

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
                category = locus_category

                if name in names:
                    raise ValueError('Loci with duplicate name found on line {}'.format(line_number))
                else:
                    names.add(name)

                locus = Locus(
                    contig=contig,
                    start=start,
                    stop=stop,
                    name=name,
                    category=category,
                    sequence=None,
                    variants=None
                )

                yield locus


def set_loci_sequence(loci, path):

    with pysam.Fastafile(path) as fasta:

        for locus in loci:

            sequence = fasta.fetch(locus.contig, locus.start, locus.stop).upper()

            locus = locus.set(sequence=sequence)

            yield locus


def set_loci_variants(loci, path):

    with pysam.VariantFile(path) as vcf:

        for locus in loci:
            variants = []

            for var in vcf.fetch(locus.contig, locus.start, locus.stop):

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


