import numpy as np

from functools import reduce
from operator import add
from collections import Counter

from haplohelper.io.genome import Locus


def _column_reference(pileupcolumn):
    ref_char = '-'
    for read in pileupcolumn.pileups:
        for _, ref_pos, char in read.alignment.get_aligned_pairs(with_seq=True):
            if ref_pos == pileupcolumn.pos:
                ref_char = char.upper()
                break
        if ref_char != '-':
            break
    return ref_char


def _column_variants(pileupcolumn, min_map_qual=0):
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
                samples[sample] = Counter(nucl)

    return samples


# functions for using the result of _column_variants

def _check_proportion(counter, threshold):
    if len(counter) < 2:
        return False
    else:
        common = reduce(max, counter.values())
        total = reduce(add, counter.values())
        if (total - common) / total >= threshold:
            return True
        else:
            return False


def _select_column_variants(samples,
                            min_mean_depth=0,
                            pop_min_proportion=0.0,
                            sample_min_proportion=0.0):
    selected = False

    totals = reduce(add, samples.values(), Counter())

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


def build_locus(alignment_file=None,
                reference=None,
                contig=None,
                start=None,
                stop=None,
                max_allele=2,
                min_map_qual=20,
                min_mean_depth=10,
                pop_min_proportion=0.1,
                sample_min_proportion=0.2,
                warn=True):

    if warn:
        Warning('This is a convenience function that should not be used in production')

    if contig is None:
        raise ValueError('contig is required')

    pileup = alignment_file.pileup(contig, start, stop)

    ref_chars = np.empty(len(range(start, stop)), dtype='U1')
    alleles = []
    positions = []

    for column in pileup:
        if column.pos in range(start, stop):
            ref_char = _column_reference(column)
            ref_chars[column.pos - start] = ref_char

            allele_counts = _column_variants(
                column,
                min_map_qual
            )
            selected = _select_column_variants(
                allele_counts,
                min_mean_depth,
                pop_min_proportion,
                sample_min_proportion
            )
            if selected:
                allele_counts = reduce(add, allele_counts.values())
                allele_counts = allele_counts.most_common(max_allele)
                allele_tuple = (ref_char, )
                allele_tuple += tuple(char for char, _ in allele_counts if char != ref_char)
                alleles.append(allele_tuple)
                positions.append(column.pos)

    sequence = ''.join(ref_chars)

    return Locus(
        reference=reference,
        contig=contig,
        start=start,
        stop=stop,
        positions=positions,
        alleles=alleles,
        sequence=sequence
    )
