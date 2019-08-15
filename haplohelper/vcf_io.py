import numpy as np

from biovector import _vector
from biovector.kmer import unique_kmer_depth as _ukmer


def argsort_genotypes(genotypes):
    """Argsort array of alleles represnted by integers.

    Each row represents genotype of ploidy alleles which
    must be sorted smallest to largest.
    """
    return np.lexsort(genotypes.T)


def sort_genotypes(genotypes):
    """Sort array of alleles represnted by integers.

    Each row represents genotype of ploidy alleles which
    must be sorted smallest to largest.
    """
    return genotypes[np.lexsort(genotypes.T)]


def allele_value_matrix(ploidy, alleles):
    array = np.zeros((alleles, ploidy), dtype=np.int64)
    array[:, 0] = np.arange(alleles)
    for i in range(1, ploidy):
        for j in range(1, alleles):
            array[j, i] = array[j - 1, i] + array[j, i - 1]
    return array


def n_genotypes(ploidy, alleles):
    return np.sum(allele_value_matrix(ploidy, alleles)[-1]) + 1


def iter_genotypes(ploidy, alleles):
    array = np.zeros(ploidy, dtype=np.int)
    yield array.copy()
    for _ in range(alleles - 1):
        for i in reversed(range(ploidy)):
            array[i] += 1
            yield array.copy()



def genotype_index(genotypes, allele_values=None):
    ploidy = genotypes.shape[-1]
    idx =  np.arange(ploidy)
    if allele_values is None:
        allele_values = allele_value_matrix(ploidy,
                                            np.max(genotypes) + 1)
    if genotypes.ndim == 1:
        return np.sum(allele_values[(genotypes, idx)])
    else:
        shape = genotypes.shape[:-1]
        n = np.prod(shape)
        return np.sum(allele_values[genotypes.ravel(),
                                    np.tile(idx, n)].reshape(n, ploidy),
                      axis=-1).reshape(shape)



def haplotypes_as_phased_snps(haplotypes):
    """Strings tot strings"""
    haplotypes.sort()
    n_snps = len(haplotypes[0])
    return ['|'.join([h[pos] for h in haplotypes]) for pos in range(n_snps)]


def vlm_to_snp_vcf_block(vlm,
                         samples,
                         posteriors,
                         reads,
                         filter_posterior_probability=0.95,
                         filter_kmer_incongruence=0.05,
                         filter_kmer_incongruence_type=3,
                         ):
    """"""
    snps = [
        {
            'CHROM': vlm.contig.strip(':'),
            'POS': str(pos),
            'REF': alleles[0],
            'ALT': ','.join(alleles[1:]),
            'ID': '.',
            'QUAL': '.',
            'FILTER': '.',
            'INFO': '.',
            'FORMAT': {},
        }
        for pos, alleles in vlm.snps
    ]

    for sample in samples:

        posterior = posteriors[sample]
        haplotypes = posterior[0][0]
        strings = [vlm.alphabet.decode(h) for h in haplotypes]
        strings.sort()
        genotypes = ['|'.join([s[pos] for s in strings])
                     for pos in range(len(snps))]

        probability = posterior[1][0]

        binary_reads=vector.binary_cast(reads[sample], 'Z')

        # phase set
        phase_set = snps[0]['POS']

        # filter based on whole haplotype
        filt = []
        if filter_posterior_probability:
            if probability < filter_posterior_probability:
                filt.append('pp<{}'.format(filter_posterior_probability))

        if filter_kmer_incongruence:
            incongruence = _ukmer(binary_reads,
                                  haplotypes,
                                  k=filter_kmer_incongruence_type,
                                  proportion=True)
            if np.any(incongruence > filter_kmer_incongruence):
                filt.append('{}mer>{}'.format(filter_kmer_incongruence_type,
                                              filter_kmer_incongruence))

        if filt:
            filt = ';'.join(filt)
        else:
            filt = 'PASS'

        # depth
        allele_depth = vector.allele_depth(binary_reads)
        depth = np.sum(allele_depth, axis=-1)

        for i in range(len(snps)):

            snps[i]['FORMAT'][sample] = {
                'GT': genotypes[i],
                'DP': str(depth[i]),
                'AD': ','.join(map(str, allele_depth[i])),
                'FT': filt,
                'PS': phase_set,
            }

    # format block
    cols = ['CHROM', 'POS', 'ID', 'REF', 'ALT','QUAL', 'FILTER', 'INFO']
    info = ['GT', 'PS', 'DP', 'AD', 'FT']

    for snp in snps:
        line = '\t'.join(snp[col] for col in cols)
        line += '\t' + ':'.join(info)
        for sample in samples:
            line += '\t' + ':'.join(snp['FORMAT'][sample][field]
                                    for field in info)

        yield line
















