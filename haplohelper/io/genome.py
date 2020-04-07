#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Locus:
    __slots__ = (
        'reference', 
        'contig', 
        'start', 
        'stop', 
        'positions', 
        'alleles', 
        'sequence'
    )
    reference: str
    contig: str
    start: int
    stop: int
    positions: list
    alleles: list
    sequence: str

    def __eq__(self, other):
        return self.as_locus_tuple() == other.as_locus_tuple()

    def __ne__(self, other):
        return self.as_locus_tuple() != other.as_locus_tuple()

    def __lt__(self, other):
        return self.as_locus_tuple() < other.as_locus_tuple()

    def __le__(self, other):
        return self.as_locus_tuple() <= other.as_locus_tuple()

    def __gt__(self, other):
        return self.as_locus_tuple() > other.as_locus_tuple()

    def __ge__(self, other):
        return self.as_locus_tuple() >= other.as_locus_tuple()

    def as_locus_tuple(self):
        chars = ''
        digits = ''
        for i in self.contig:
            if i.isdigit():
                digits += i
            else:
                chars += i
        return chars, int(digits), self.start, self.stop

    @property
    def range(self):
        return range(self.start, self.stop)

    def as_dict(self):
        return {slot: getattr(self, slot) for slot in self.__slots__}

    def count_alleles(self):
        return [len(tup) for tup in self.alleles]


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


def format_haplotype(locus, alleles, gap='N'):
    """Format integer encoded alleles as a haplotype string"""
    variants = (locus.alleles[i][a] if a >= 0 else gap for i, a in enumerate(alleles))
    return _template_sequence(locus).format(*variants)


def write_loci(loci, path, loci_type='HaplotypeInterval'):
    columns = (
        'CHROM',
        'START',
        'STOP',
        'TYPE',
        'NAME',
        'REF',
        'ALT',
        'WITHIN',
    )

    with open(path, 'w') as f:

        # write header line:
        f.write('#HaploHelper Loci Format v0.01\n')
        f.write('#' + '\t'.join(columns) + '\n')

        for locus in loci:

            locus_name = '{}:{}-{}'.format(locus.contig, locus.start, locus.stop)

            # write locus line:
            line = '\t'.join([
                locus.contig,
                str(locus.start),
                str(locus.stop),
                loci_type,
                locus_name,
                locus.sequence if locus.sequence else '.',
                '.',
                '.',
                '\n'
            ])
            f.write(line)
            
            for position, alleles in zip(locus.positions, locus.alleles):

                # write variant line:
                line = '\t'.join([
                    locus.contig,
                    str(position),
                    str(position + 1),  # interval of length 1 for snp
                    'SNP',
                    '.',
                    alleles[0],
                    ','.join(alleles[1:]),
                    locus_name,
                    '\n'
                ])
                f.write(line)


def read_loci(path, loci_type='HaplotypeInterval', skip_non_variable=True):

    loci = {}

    with open(path, 'r') as f:
        for line_number, line in enumerate(f):
            
            if line[0] == '#':
                pass
            
            else:
                line = line.split()

                if line[3] == loci_type:
                    # this is a locus line

                    locus = Locus(
                        reference=None,
                        contig=line[0],
                        start=int(line[1]),
                        stop=int(line[2]),
                        positions=[],
                        alleles=[],
                        sequence=line[5]
                    )

                    locus_name = line[4]

                    if locus_name is '.':
                        raise IOError('Unnamed Locus on line {}'.format(line_number))
                    
                    elif locus_name in loci:
                        raise IOError('Duplicate Locus on line {}'.format(line_number))

                    loci[locus_name] = locus
                
                else:
                    # assume this is a variant line

                    locus_name = line[7]

                    if locus_name is '.':
                        raise IOError('Variant without locus on line {}'.format(line_number))

                    elif locus_name not in loci:
                        raise IOError('Variant on line {} found before Locus "{}"'.format(line_number, locus_name))

                    pos=int(line[1])
                    ref = line[5]
                    alts = line[6]
                    alleles = (ref + ',' + alts).split(',')

                    locus_interval = loci[locus_name].range
                    if pos not in locus_interval:
                        raise IOError('Variant on line {} not within Locus "{}"'.format(line_number, locus_name))

                    loci[locus_name].positions.append(pos)
                    loci[locus_name].alleles.append(alleles)

    if skip_non_variable:
        loci = [locus for locus in loci.values() if len(locus.positions) > 0]
    else:
        loci = list(loci.values())
    return loci



