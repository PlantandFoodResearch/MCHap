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


def format_haplotype(locus, alleles, gap='N'):
    """Format integer encoded alleles as a haplotype string"""
    variants = (locus.alleles[i][a] if a >= 0 else gap for i, a in enumerate(vector))
    return locus.sequence.format(*variants)

