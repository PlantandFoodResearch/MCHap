#!/usr/bin/env python3

import numpy as np
from collections import Counter as _Counter
from functools import reduce as _reduce
from operator import add as _add
import biovector as bv
from haplohelper import util


def _count_column_variants(pileupcolumn,
                           min_map_qual=0):
    samples = dict()

    for pileupread in pileupcolumn.pileups:
        if (not pileupread.is_del
                and not pileupread.is_refskip
                and not pileupread.alignment.is_duplicate
                and pileupread.alignment.mapq >= min_map_qual):
            sample = pileupread.alignment.get_tag('RG')
            nucl = pileupread.alignment.query_sequence[
                pileupread.query_position]

            if sample in samples:
                samples[sample].update(nucl)
            else:
                samples[sample] = _Counter(nucl)

    return samples


def _check_proportion(counter, threshold):
    if len(counter) < 2:
        return False
    else:
        common = _reduce(max, counter.values())
        total = _reduce(_add, counter.values())
        if (total - common) / total >= threshold:
            return True
        else:
            return False


def _select_column(samples,
                   min_mean_depth=0,
                   pop_min_proportion=0.0,
                   sample_min_proportion=0.0):
    selected = False

    totals = _reduce(_add, samples.values(), _Counter())

    if len(totals) > 1:
        depths = [sum(sample.values()) for sample in samples.values()]

        if np.mean(depths) >= min_mean_depth:

            if _check_proportion(totals, pop_min_proportion):
                selected = True
            else:
                variable = [_check_proportion(sample, sample_min_proportion)
                            for sample in samples.values()]
                if any(v and (d >= min_mean_depth) for v, d in zip(variable,
                                                                   depths)):
                    selected = True

    return selected


def _extract_column(pileupcolumn, min_map_qual=0):

    for pileupread in pileupcolumn.pileups:
        if (not pileupread.is_del
                and not pileupread.is_refskip
                and not pileupread.alignment.is_duplicate
                and pileupread.alignment.mapq >= min_map_qual):

            sample = pileupread.alignment.get_tag('RG')
            qname = pileupread.alignment.qname
            char = pileupread.alignment.query_sequence[
                pileupread.query_position]
            qual = pileupread.alignment.query_qualities[
                pileupread.query_position]

            yield sample, qname, char, qual


def _encode_variant_data(positions, variants, alphabet):
    n_nucl = alphabet.vector_size()
    n_base = len(positions)
    pos_map = dict(zip(positions, range(len(positions))))
    arrays = dict()

    for name, sample in variants.items():
        n_reads = len(sample)
        array = np.zeros((n_reads, n_base, n_nucl),
                         dtype=np.float64)
        array[:] = 1 / alphabet.vector_size()
        for i, read in enumerate(sample.values()):
            for pos, (char, qual) in read.items():
                prob = util.prob_of_qual(qual)
                array[i, pos_map[pos]] = alphabet.encode_element(char, prob)
        arrays[name] = array

    return arrays


def find_variants(alignment_file,
                  contig,
                  start,
                  stop,
                  min_map_qual=20,
                  min_mean_depth=10,
                  pop_min_proportion=0.1,
                  sample_min_proportion=0.2,
                  n_alleles=2):

    positions = list()
    interval = range(start, stop)
    totals = list()

    for pileupcolumn in alignment_file.pileup(contig, start, stop):
        if pileupcolumn.pos not in interval:
            pass
        else:
            counts = _count_column_variants(pileupcolumn,
                                            min_map_qual)

            selected = _select_column(counts,
                                      min_mean_depth,
                                      pop_min_proportion,
                                      sample_min_proportion)

            if selected:
                positions.append(pileupcolumn.pos)
                totals.append(_reduce(_add, counts.values(), _Counter()))

    return VariantLociMap.from_counts(
        totals,
        reference_positions=positions,
        reference_name=alignment_file.reference_filename,
        reference_contig=contig,
        n_alleles=n_alleles
    )


def encode_alignment_positions(alignment_file,
                               contig=None,
                               positions=None,
                               alphabet=None,
                               min_map_qual=20):

    # extract reads

    # sample: read: array of variants with phred scores
    reads = dict()

    # maps reference position to array index
    pos_map = dict(zip(positions, range(len(positions))))

    # variants stored in array
    # default values for read gaps is a null allele 'N' with probability 1
    dtype_variant = np.dtype([('char', np.str_, 1), ('prob', np.float)])
    template = np.empty(len(positions), dtype=dtype_variant)
    template['char'] = 'N'
    template['prob'] = 1.0

    for pileupcolumn in alignment_file.pileup(contig,
                                              np.min(positions),
                                              np.max(positions)):
        if pileupcolumn.pos in pos_map:
            pos = pileupcolumn.pos

            for sample, qname, char, qual in _extract_column(pileupcolumn,
                                                             min_map_qual):
                if sample not in reads:
                    reads[sample] = {}
                if qname not in reads[sample]:
                    reads[sample][qname] = template.copy()
                reads[sample][qname][pos_map[pos]] = (
                    char,
                    util.prob_of_qual(qual)
                )

    # encode reads (reuse same dict)
    for sample, data in reads.items():
        array = np.empty((len(data), len(positions), alphabet.vector_size()),
                         dtype=np.float)
        for i, read in enumerate(data.values()):

            array[i] = alphabet.encode(read)

        reads[sample] = array

    return reads


class VariantLociMap(object):
    """Block of variant positions
    """

    def __init__(self,
                 iupac_to_allelic,
                 reference_positions=None,
                 reference_name=None,
                 reference_contig=None,
                 n_alleles=2):

        self._iupac_to_allelic = iupac_to_allelic

        if reference_positions is None:
            reference_positions = np.repeat(np.nan, len(iupac_to_allelic))
        self.reference_positions = reference_positions

        self.reference_name = reference_name
        self.reference_contig = reference_contig

        assert n_alleles in {2, 3, 4}
        self.n_alleles = n_alleles
        if self.n_alleles == 2:
            self.alphabet = bv.Biallelic
        elif self.n_alleles == 3:
            self.alphabet = bv.Triallelic
        elif self.n_alleles == 4:
            self.alphabet = bv.Quadraallelic

        self._allelic_to_iupac = np.empty(len(self._iupac_to_allelic),
                                          dtype='<O')
        alleles = set(map(str, range(self.n_alleles)))
        for i, d in enumerate(self._iupac_to_allelic):
            self._allelic_to_iupac[i] = {'N': 'N'}

            for k, v in d.items():
                if v in alleles:
                    self._allelic_to_iupac[i][v] = k

    def __len__(self):
        return len(self._iupac_to_allelic)

    def __getitem__(self, item):
        if isinstance(item, int):
            item = slice(item, item + 1)

        if self.reference_positions is None:
            reference_positions = None
        else:
            reference_positions = self.reference_positions[item]

        return VariantLociMap(self._iupac_to_allelic[item],
                              reference_positions,
                              reference_name=self.reference_name,
                              reference_contig=self.reference_contig,
                              n_alleles=self.n_alleles)

    def __repr__(self):
        header = '{0}\n{1}\nPOS\tREF\tALT\n'.format(self.reference_name,
                                                    self.reference_contig)
        zipped = zip(self.reference_positions,
                     self.reference_alleles,
                     self.alternate_alleles)
        data = '\n'.join('{0}\t{1}\t{2}'.format(*tup) for tup in zipped)
        return header + data

    def vector_size(self):
        return self.alphabet.vector_size()

    @classmethod
    def from_counts(cls, counts, *args, **kwargs):
        # assumes most common allele is reference (for now)
        n_alleles = kwargs['n_alleles']
        iupac_to_allelic = np.empty(len(counts), dtype='<O')
        for i, counter in enumerate(counts):

            iupac_to_allelic[i] = {k: 'N' for k in 'ACTGN'}

            j = 0
            for char, _ in counter.most_common(n_alleles):
                # need to handle N as an allele
                if char == 'N':
                    pass
                else:
                    iupac_to_allelic[i][char] = str(j)
                    j += 1

        return cls(iupac_to_allelic, *args, **kwargs)

    def as_allelic(self, string):
        assert len(self) == len(string)
        return ''.join(
            self._iupac_to_allelic[i][char] for i, char in enumerate(string))

    def as_iupac(self, string):
        assert len(self) == len(string)
        return ''.join(
            self._allelic_to_iupac[i][char] for i, char in enumerate(string))

    def decode(self, array):
        return self.as_iupac(self.alphabet.decode(array))

    def encode(self, data):
        assert len(data) == len(self)

        data_length = len(data[0])
        if data_length == 1:
            # binary encoding
            dtype = self.alphabet.dtype()
        elif data_length == 2:
            # probabilistic encoding
            dtype = np.float
        else:
            raise ValueError('data must be a sequence of characters or '
                             'pairs of characters with probabilities.')
        array = np.empty((len(data), self.vector_size()), dtype=dtype)
        for i, symbol in enumerate(data):
            symbol = tuple(symbol)
            array[i] = self.alphabet.encode_element(
                self._iupac_to_allelic[i][symbol[0]],
                *symbol[1:]
            )
        return array

    def encode_alignment(self, alignment_file, min_map_qual=20):
        return encode_alignment_positions(alignment_file,
                                          alphabet=self,
                                          contig=self.reference_contig,
                                          positions=self.reference_positions,
                                          min_map_qual=min_map_qual)

    @property
    def reference_alleles(self):
        return self.as_iupac('0' * len(self))

    @property
    def alternate_alleles(self):
        return tuple(','.join(k for k, v in d.items()
                              if v not in '0N')
                     for d in self._iupac_to_allelic)
