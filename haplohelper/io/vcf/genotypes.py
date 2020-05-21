from collections import Counter
from dataclasses import dataclass

from haplohelper.io import format_haplotypes


@dataclass(frozen=True)
class Genotype(object):
    alleles: tuple
    phased: bool = False

    def __str__(self):
        sep = '|' if self.phased else '/'
        return sep.join((str(i) if i >= 0 else '.' for i in self.alleles))


def label_haplotypes(locus, genotypes): # genotypes is a dict
    
    # turn genotype arrays to haplotype strings
    samples = {
        sample: format_haplotypes(locus, array) 
        for sample, array in genotypes.items()
    }
    
    # counts of alleles amoung all samples
    counts = Counter()
    for strings in samples.values():
        counts += Counter(strings)
    
    # ensure reference is included and is first
    if locus.sequence not in counts:
        ref_count = 0
    else:
        ref_count = counts.pop(locus.sequence)
    tups = counts.most_common()
    tups = [(locus.sequence, 0)] + tups

    # alleles with counts with ref first
    alleles, counts = tuple(zip(*tups))
    
    # now label alleles in each sample
    numbers = {a: i for i, a in enumerate(alleles)}
        
    genotypes = {
        sample: Genotype(tuple(numbers[s] for s in strings))
        for sample, strings in samples.items()
    }
    
    return alleles, counts, genotypes
