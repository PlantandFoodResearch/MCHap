#!/usr/bin/env python3

from dataclasses import dataclass


@dataclass
class Locus:
    __slots__ = (
        'reference', 
        'contig', 
        'interval', 
        'positions', 
        'alleles', 
        'sequence'
    )
    reference: str
    contig: str
    interval: range
    positions: list
    alleles: list
    sequence: str


def _template_sequence(locus):
    chars = list(locus.sequence)
    ref_alleles = (tup[0] for tup in locus.alleles)
    for pos, string in zip(locus.positions, ref_alleles):
        idx = pos - locus.interval.start
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

