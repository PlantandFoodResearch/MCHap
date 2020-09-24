import numpy as np
from collections import Counter
from dataclasses import dataclass
from functools import reduce
from itertools import combinations_with_replacement

from mchap import mset


def _allele_sort(alleles):
    """Sort alleles for VCF output, null alleles follow called alleles.
    """
    calls = [i for i in alleles if i >= 0]
    nulls = [i for i in alleles if i < 0]
    calls.sort()
    return tuple(calls + nulls)


def call_best_genotype(genotypes, probabilities):
    """Returns the genotype with highest probability.
    """
    assert len(genotypes) == len(probabilities)
    idx = np.argmax(probabilities)
    return genotypes[idx], probabilities[idx]


def call_phenotype(genotypes, probabilities, threshold=0.95):
    """Identifies the most complete set of alleles that 
    exceeds a probability threshold.
    If the probability threshold cannot be exceeded the
    phenotype will be returned with a probability of None
    """
    assert len(genotypes) == len(probabilities)

    # check mode genotype
    if np.max(probabilities) >= threshold:
        # mode genotype prob greater than threshold
        idx = np.argmax(probabilities)
        return genotypes[idx], probabilities[idx]
    
    # result will require some padding with null alleles
    _, ploidy, n_pos = genotypes.shape
    result = np.zeros((ploidy, n_pos), dtype=genotypes.dtype) - 1

    # find genotype combination that meets threshold
    selected = list()
    p = 0.0
    genotypes = list(genotypes)
    probabilities = list(probabilities)
    
    while p < threshold:
        if len(probabilities) == 0:
            # all have been selected
            break
        idx = np.argmax(probabilities)
        p += probabilities.pop(idx)
        selected.append(genotypes.pop(idx))

    # intercept of selected genotypes
    alleles = reduce(mset.intercept, selected)
    
    # result adds padding with null alleles
    for i, hap in enumerate(alleles):
        result[i] = hap
    
    return result, p


@dataclass(order=True, frozen=True)
class Genotype(object):
    alleles: tuple
    phased: bool = False

    def __str__(self):
        sep = '|' if self.phased else '/'
        return sep.join((str(i) if i >= 0 else '.' for i in self.alleles))

    def sorted(self):
        if self.phased:
            # can't sort phased data
            raise ValueError('Cannot sort a phased genotype.')
        else:
            # values < 0 must come last
            alleles = _allele_sort(self.alleles)
            return type(self)(alleles)


@dataclass
class HaplotypeAlleleLabeler(object):
    alleles: tuple

    @classmethod
    def from_obs(cls, genotypes):
        """Construct a new instance with alternate alleles ordered
        by frequency among observed genotypes (highest to lowest).
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

        alleles = tuple(labels.get(tuple(hap), -1) for hap in array)
        if not phased:
            alleles = _allele_sort(alleles)
        
        return Genotype(alleles)

    def label_phenotype_posterior(self, array, probs, unobserved=True, expected_dosage=False):
        """Create a VCF genotype from one or more genotype arrays
        that share the same alleles but with different dosage.
        """
        n_gen, ploidy, n_pos = np.shape(array)
        assert len(array) == len(probs)

        if np.all(array < 0):
            # null alleles
            genotypes = [Genotype(tuple(-1 for _ in range(ploidy)))]
            probs = [None]
            if not expected_dosage:
                return genotypes, probs
            else:
                return genotypes, probs, [None]

        # label observed genotypes
        observed = [self.label(gen) for gen in array]

        # alleles observed in first genotype
        alleles = set(observed[0].alleles)

        # these should be identical in all observed genotypes
        assert all(alleles == set(gen.alleles) for gen in observed)

        # phenotype as a sorted tuple
        alleles = _allele_sort(alleles)

        # all possible genotypes for this phenotype with prob of 0
        genotypes = {}
        
        # optionally include unobserved genotypes
        if unobserved:
            opts = combinations_with_replacement(alleles, ploidy - len(alleles))
            for opt in opts:
                opt = _allele_sort(alleles + opt)
                opt = Genotype(opt)
                genotypes[opt] = 0.0

        # add observed probs
        for obs, prob in zip(observed, probs):
            genotypes[obs] = prob

        # unpack dict
        genotypes, probs = zip(*genotypes.items())

        # sort by genotypes
        idx = np.argsort(genotypes)
        genotypes = [genotypes[i] for i in idx]
        probs = [probs[i] for i in idx]

        if not expected_dosage:
            return genotypes, probs

        # normalised probs
        normalised = np.array(probs) / np.sum(probs)

        # dosage weighted by prob of each genotype
        dosage_exp = np.zeros(len(alleles), dtype=np.float)
        for gen, p in zip(genotypes, normalised):
            counts = Counter(gen.alleles)
            for i, a in enumerate(alleles):
                dosage_exp[i] += counts[a] * p

        return genotypes, probs, list(dosage_exp)


    def ref_array(self):
        """Returns reference allele array.
        """
        return np.array(self.alleles[0], dtype=np.int8)

    def alt_array(self):
        """Returns array of alternate alleles.
        """
        return np.array(self.alleles[1:], dtype=np.int8)
