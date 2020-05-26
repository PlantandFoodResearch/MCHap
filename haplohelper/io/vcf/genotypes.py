import numpy as np
from collections import Counter
from dataclasses import dataclass
from itertools import combinations_with_replacement

from haplohelper.io import format_haplotypes


@dataclass(order=True, frozen=True)
class Genotype(object):
    alleles: tuple
    phased: bool = False

    def __str__(self):
        sep = '|' if self.phased else '/'
        return sep.join((str(i) if i >= 0 else '.' for i in self.alleles))


@dataclass
class HaplotypeAlleleLabeler(object):
    alleles: tuple

    @classmethod
    def from_obs(cls, genotypes):
        """Construct a new instance with alternate alleles ordered
        by frequency amoung observed genotypes (highest to lowest).
        """
        haplotypes = np.concatenate(genotypes)
        _, n_pos = haplotypes.shape
        
        counts = Counter(tuple(hap) for hap in haplotypes)
        ref = (0, ) * n_pos
        null = (-1, ) * n_pos

        if null in counts:
            _ = counts.pop(null)

        if ref not in counts:
            ref_count = 0
        else:
            ref_count = counts.pop(ref)

        pairs = counts.most_common()
        pairs = [(ref, ref_count)] + pairs

        alleles = tuple(a for a, _ in pairs)

        return cls(alleles)

    def count_obs(self, genotypes):
        """Counts frequency of alleles amoung observed genotypes 
        """
        haplotypes = np.concatenate(genotypes)
        
        labels = {a: i for i, a in enumerate(self.alleles)}
        counts = np.zeros(len(self.alleles), dtype=np.int)
        for hap in haplotypes:
            if np.all(hap < 0):
                # null allele
                pass
            else:
                allele = labels[tuple(hap)]
                counts[allele] += 1
        return counts
    
    def label(self, array, phased=False):
        """Create a VCF genotype from a genotype array.
        """

        labels = {a: i for i, a in enumerate(self.alleles)}

        alleles = [labels.get(tuple(hap), -1) for hap in array]
        if not phased:
            alleles.sort()
        
        return Genotype(tuple(alleles))

    def label_phenotype_posterior(self, array, probs, unobserved=True):
        """Create a VCF genotype from one or more genotype arrays
        that share the same alleles but with different dosage.
        """
        n_gen, ploidy, n_pos = np.shape(array)
        assert len(array) == len(probs)

        # label observed genotypes
        observed = [self.label(gen) for gen in array]

        # alleles observed in first genotype
        alleles = set(observed[0].alleles)

        # these should be identical in all observed genotypes
        assert all(alleles == set(gen.alleles) for gen in observed)

        # phenotype
        alleles = list(alleles)
        alleles.sort()
        alleles = tuple(alleles)
        #nulls = tuple(-1 for _ in range(ploidy - len(alleles)))
        #phenotype = Genotype(alleles + nulls)

        # all possible genotypes for this phenotype with prob of 0
        genotypes = {}
        
        # optionally include unobserved genotypes
        if unobserved:
            opts = combinations_with_replacement(alleles, ploidy - len(alleles))
            for opt in opts:
                t = alleles + opt
                l = list(t)
                l.sort()
                g = Genotype(tuple(l))
                genotypes[g] = 0.0

        # add observed probs
        for obs, prob in zip(observed, probs):
            genotypes[obs] = prob

        # unpack dict
        genotypes, probs = zip(*genotypes.items())

        # sort by genotypes
        idx = np.argsort(genotypes)
        genotypes = [genotypes[i] for i in idx]
        probs = [probs[i] for i in idx]

        return genotypes, probs

    def ref_array(self):
        """Returns reference allele array.
        """
        return np.array(self.alleles[0], dtype=np.int8)

    def alt_array(self):
        """Returns array of alternate alleles.
        """
        return np.array(self.alleles[1:], dtype=np.int8)
