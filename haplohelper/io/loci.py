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


class LociFile(pysam.Tabixfile):
    
    @property
    def header(self):
        data = {
            'LOCI': set(),
            'VARIANTS': set(),
        }
        
        for line in super().header:
            if line.startswith('#LOCI='):
                line = line.split('=')[1]
                data['LOCI'] |= {string.strip() for string in line.split(';')}
            elif line.startswith('#VARIANTS='):
                line = line.split('=')[1]
                data['VARIANTS'] |= {string.strip() for string in line.split(';')}
            
        return data
    
    def fetch(self, *args, **kwargs):
        
        header_data = self.header
        loci_types = header_data['LOCI']
        variant_types = header_data['VARIANTS']
        
        names = set()
        
        lines = super().fetch(*args, **kwargs)
        
        locus = None
        
        for line in lines:
                        
            line = line.split()
            
            if line[4] in loci_types:
                
                
                if locus:
                    # yield the previous locus with its variants
                    yield locus.set(variants=tuple(variants))
                
                else:
                    pass
                
                # create a locus without variants
                locus = Locus(
                    contig=line[0],
                    start=int(line[1]),
                    stop=int(line[2]),
                    name=line[3],
                    category=line[4],
                    variants=None,
                    sequence=line[5]
                )
                
                # new list to collect variants
                variants = []
                                
                if locus.name is '.':
                    raise IOError('Unnamed Locus at "{}:{}-{}"'.format(
                        locus['contig'],
                        locus['start'],
                        locus['stop']
                    ))
                
                elif locus.name in names:
                    raise IOError('Duplicate Locus named "{}"'.format(locus.name))
                    
                names.add(locus.name)
                
            if line[4] in variant_types:
                
                var = Variant(
                    contig=line[0],
                    start=int(line[1]),
                    stop=int(line[2]),
                    name=line[3],
                    category=line[4],
                    alleles=tuple((line[5] + ',' + line[6]).split(',')),
                )
                
                # name for reporting errors
                name = var.name if var.name not in {'.', None} else '{}:{}-{}'.format(var.contig, var.start, var.stop)
                
                parent = line[7]
                
                # check theat there is a locus 
                if not locus:
                    raise IOError('Variant "{}" found before locus "{}"'.format(name, parent))
                
                # check that variant matches the current locus
                elif locus.name != parent:
                    
                    if parent in names:
                        raise IOError('Variant "{}" found after locus "{}"'.format(name, parent))
                        
                    else:
                        raise IOError('Variant "{}" found before locus "{}"'.format(name, parent))
                        
                # check that variant is within the locus interval
                if var.start not in locus.range:
                    raise IOError('Variant "{}" does not fall within locus "{}"'.format(name, parent))
                    
                # append variant to list started for the locus
                variants.append(var)
        
        if locus:
            # yield the final locus with its variants
            yield locus.set(variants=tuple(variants))
