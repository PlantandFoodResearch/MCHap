import numpy as np
from haplohelper import mset
from haplohelper.io import format_haplotypes as _format_haplotypes


def label_haplotypes(genotypes):
    haplotypes = np.concatenate(genotypes)
    categories, _ = mset.unique_counts(haplotypes, order='descending')
    
    # mask out ref allele (if present)
    mask = ~np.all(categories == 0, axis=-1)
    categories = categories[mask]
    
    # add ref allele in first position
    ref_allele = np.zeros((1, categories.shape[-1]), categories.dtype)
    categories = np.concatenate([ref_allele, categories])

    return [mset.categorize(haps, categories) for haps in genotypes]


def info_col(**kwargs):
    """Takes arguments mapped to strings and formats as a VCF "INFO" field.
    """
    return ';'.join(('{}={}'.format(k, str(v)) for k, v in kwargs.items()))


def format_cols(**kwargs):
    """Takes arguments mapped to lists of strings and formats as a VCF "FORMAT" field.
    Lists should be of equal length and items within lists should be in order of samples.
    """
    fields = list(kwargs.keys())
    string = ':'.join(fields)
    lists = [kwargs[key] for key in fields]
    samples = [':'.join(tup) for tup in zip(*lists)]
    return string, samples


def vcf_line(
        chrom='.',
        pos='.',
        id='.',
        ref='.',
        alt='.',
        qual='.',
        filter='.',
        info='.',
        format='.',
        samples=None
    ):
    cols = [
        chrom,
        str(pos) if isinstance(pos, int) else pos,
        id,
        ref,
        alt,
        str(qual) if isinstance(qual, int) else qual,
        filter,
        info,
        format,
    ]
    if samples:
        cols += list(samples)
    
    return '\t'.join(cols) + '\n'


def col_REF_haplotypes(locus):
    return locus.sequence


def col_ALT_haplotypes(locus, genotypes):
    # genotypes are the raw genotypes
    haplotypes = np.concatenate(genotypes)
    alts, _ = mset.unique_counts(haplotypes, order='descending')
    
    # mask out ref allele (if present)
    mask = ~np.all(alts == 0, axis=-1)
    alts = alts[mask]

    if np.size(alts) == 0:
        # special case if there are no alts
        return '.'
    else:
        return ','.join(_format_haplotypes(locus, alts))


def col_CHROM(locus):
    return locus.contig


def col_POS(locus):
    return(str(locus.start))


def info_VP(locus):
    return ','.join(str(pos - locus.start) for pos in locus.positions)


def info_NS(genotypes):
    return str(len(genotypes))


def info_END(locus):
    return str(locus.stop)


def info_AC(genotypes):
    cats, counts = np.unique(np.concatenate(genotypes), return_counts=True)
    return ','.join(map(str, counts[np.argsort(cats)]))


def info_AN(genotypes):
    return str(len(np.unique(np.concatenate(genotypes))))


def format_GT(vectors, phased=False):
    sep = '|' if phased else '/'
    return [sep.join(map(str, np.sort(vec))) for vec in vectors]


def haplotype_line(locus, genotypes, **kwargs):
    """Kwargs are interpreted as aditional format lines.
    """
    haplotypes = label_haplotypes(genotypes)


    info = info_col(
        NS=info_NS(haplotypes),
        AN=info_AN(haplotypes),
        AC=info_AC(haplotypes),
        END=info_END(locus),
        VP=info_VP(locus),
    )

    format, samples = format_cols(
        GT=format_GT(haplotypes),
        **kwargs
    )

    return vcf_line(
        chrom=col_CHROM(locus),
        pos=col_POS(locus),
        ref=col_REF_haplotypes(locus),
        alt=col_ALT_haplotypes(locus, genotypes),
        info=info,
        format=format,
        samples=samples,
    )

