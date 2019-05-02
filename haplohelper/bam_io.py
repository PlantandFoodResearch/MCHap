#!/usr/bin/env python3

import numpy as np
from collections import Counter as _Counter
from functools import reduce as _reduce
from operator import add as _add

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
                  interval,
                  min_map_qual=20,
                  min_mean_depth=10,
                  pop_min_proportion=0.1,
                  sample_min_proportion=0.2):

    positions = list()
    interval = range(*interval)

    for pileupcolumn in alignment_file.pileup(contig,
                                              interval.start,
                                              interval.stop):
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
    return positions


def encode_variants(alignment_file,
                    contig,
                    interval,
                    alphabet,
                    min_map_qual=20,
                    min_mean_depth=10,
                    pop_min_proportion=0.1,
                    sample_min_proportion=0.2):

    positions = list()
    variants = dict()
    interval = range(*interval)

    for pileupcolumn in alignment_file.pileup(contig,
                                              interval.start,
                                              interval.stop):
        if pileupcolumn.pos in interval:
            counts = _count_column_variants(pileupcolumn,
                                            min_map_qual)

            selected = _select_column(counts,
                                      min_mean_depth,
                                      pop_min_proportion,
                                      sample_min_proportion)

            if selected:
                pos = pileupcolumn.pos
                positions.append(pos)

                for sample, qname, char, qual in _extract_column(pileupcolumn,
                                                                 min_map_qual):
                    if sample not in variants:
                        variants[sample] = {}
                    if qname not in variants[sample]:
                        variants[sample][qname] = {}
                    variants[sample][qname][pos] = char, qual

    arrays = _encode_variant_data(positions, variants, alphabet)

    return positions, arrays


def encode_positions(alignment_file,
                     contig,
                     positions,
                     alphabet,
                     min_map_qual=20):
    variants = dict()
    pos_map = dict(zip(positions, range(len(positions))))

    for pileupcolumn in alignment_file.pileup(contig,
                                              np.min(positions),
                                              np.max(positions)):
        if pileupcolumn.pos in pos_map:
            pos = pileupcolumn.pos

            for sample, qname, char, qual in _extract_column(pileupcolumn,
                                                             min_map_qual):
                if sample not in variants:
                    variants[sample] = {}
                if qname not in variants[sample]:
                    variants[sample][qname] = {}
                variants[sample][qname][pos] = char, qual

    return _encode_variant_data(positions, variants, alphabet)
