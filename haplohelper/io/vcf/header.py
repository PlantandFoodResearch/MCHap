from datetime import date as _date

_FILTER_HEADER_TEMPLATE = '##FILTER=<ID={code},Description="{desc}">\n'

_INFO_HEADER_TEMPLATE = '##INFO=<ID={id},Number={number},Type={type},Description="{desc}">\n'

_INFO_HEADERS = {
    'NS': _INFO_HEADER_TEMPLATE.format(id='NS', number=1, type='Integer', desc='Number of samples with data'),
    'DP': _INFO_HEADER_TEMPLATE.format(id='DP', number=1, type='Integer', desc='Combined depth across samples'),
    'AC': _INFO_HEADER_TEMPLATE.format(id='AC', number='A', type='Integer', desc='Allele count in genotypes, for each ALT allele, in the same order as listed'),
    'AN': _INFO_HEADER_TEMPLATE.format(id='AN', number=1, type='Integer', desc='Total number of alleles in called genotypes'),
    'AF': _INFO_HEADER_TEMPLATE.format(id='AF', number='A', type='Float', desc='Allele Frequency'),
    'AA': _INFO_HEADER_TEMPLATE.format(id='AA', number=1, type='String', desc='Ancestral allele'),
    'END': _INFO_HEADER_TEMPLATE.format(id='END', number=1, type='Integer', desc='End position on CHROM'),
    'VP': _INFO_HEADER_TEMPLATE.format(id='VP', number='.', type='Integer', desc='Relative positions of SNPs within haplotypes'),
}


_FORMAT_HEADER_TEMPLATE = '##FORMAT=<ID={id},Number={number},Type={type},Description="{desc}">\n'

_FORMAT_HEADERS = {
    'GT': _FORMAT_HEADER_TEMPLATE.format(id='GT', number=1, type='String', desc='Genotype'),
    'GQ': _FORMAT_HEADER_TEMPLATE.format(id='GQ', number=1, type='Integer', desc='Genotype quality'),
    'DP': _FORMAT_HEADER_TEMPLATE.format(id='DP', number=1, type='Integer', desc='Read depth'),
    'PS': _FORMAT_HEADER_TEMPLATE.format(id='PS', number=1, type='Integer', desc='Phase set'),
    'FT': _FORMAT_HEADER_TEMPLATE.format(id='FT', number=1, type='String', desc='Filter indicating if this genotype was called'),
    'GP': _FORMAT_HEADER_TEMPLATE.format(id='GP', number='G', type='Float', desc='Genotype posterior probabilities'),
}


def file_format_line():
    return '##fileformat=VCFv4.3\n'


def file_date_line():
    d = _date.today()
    return '##fileDate={}{}{}\n'.format(d.year, d.month, d.day)


def source_line(version='v0.0.1'):
    return '##source=HaploKit-{}\n'.format(version)


def reference_line(path):
    return '##reference=file:{}\n'.format(path)


def contig_line(id, **kwargs):
    line = ['##contig=<ID={}'.format(id)]
    line += ['{}={}'.format(k, v) for k, v in kwargs.items()]
    return ','.join(line) + '\n'


def phasing_line(string):
    return '##phasing={}\n'.format(string)


def info_line(code):
    return _INFO_HEADERS[code]


def format_line(code):
    return _FORMAT_HEADERS[code]


def kmer_filter_line(k=3, threshold=0.95):
    description = 'Less than {} % of samples read-variant {}-mers '.format(threshold * 100, k)
    code = 'k{}<{}'.format(k, threshold)
    return _FILTER_HEADER_TEMPLATE.format(code=code, desc=description)


def depth_filter_line(threshold=5.0):
    description = 'Sample has mean read depth less than {}.'.format(threshold)
    code = 'd<{}'.format(threshold)
    return _FILTER_HEADER_TEMPLATE.format(code=code, desc=description)


def prob_filter_line(threshold=0.95):
    description = 'Samples genotype posterior probability < {}.'.format(threshold)
    code = 'p<{}'.format(threshold)
    return _FILTER_HEADER_TEMPLATE.format(code=code, desc=description)

def body_header_line(samples):

    cols = [
        '#CHROM',
        'POS',
        'ID',
        'REF',
        'ALT',
        'QUAL',
        'FILTER',
        'INFO',
        'FORMAT'
    ]

    cols += list(samples)

    return '\t'.join(cols) + '\n'


def haplotype_header(loci, samples, info_codes, format_codes, filter_lines=None):

    lines = [
        file_format_line(),
        file_date_line(),
        source_line(),
        reference_line('.'),
        phasing_line('None'),
    ]

    lines += [contig_line(name) for name in {locus.contig for locus in loci}]

    lines += [info_line(code) for code in info_codes]

    if filter_lines:
        lines += filter_lines

    lines += [format_line(code) for code in format_codes]

    lines += [body_header_line(samples)]

    return lines






