import numpy as np
from mchap.io.vcf.util import vcfstr


def format_info_field(**kwargs):
    """Format key-value pairs into a VCF info field.

    Parameters
    ----------
    kwargs
        Key value pairs of info field codes to values.

    Returns
    -------
    string : str
        VCF info field.
    """
    parts = ["{}={}".format(k, vcfstr(v)) for k, v in kwargs.items()]
    return ";".join(parts)


def format_sample_field(**kwargs):
    """Format key-value pairs into a VCF format field.

    Parameters
    ----------
    kwargs
        Key value pairs of info field codes to arrays of values per sample.

    Returns
    -------
    string : str
        VCF format and sample columns.
    """
    fields, arrays = zip(*kwargs.items())
    fields = ":".join(fields)
    lengths = np.array([len(a) for a in arrays])
    length = lengths[0]
    assert np.all(lengths == length)
    sample_data = np.empty(length, dtype="O")
    for i in range(length):
        sample_data[i] = ":".join((vcfstr(a[i]) for a in arrays))
    sample_data = "\t".join(sample_data)
    return "{}\t{}".format(fields, sample_data)


def format_record(
    *,
    chrom=None,
    pos=None,
    id=None,
    ref=None,
    alt=None,
    qual=None,
    filter=None,
    info=None,
    format=None,
):
    """Format a VCF record line.

    Parameters
    ----------
    chrom : str
        Variant chromosome or contig.
    pos : int
        Variant position.
    id : str
        Variant ID.
    ref : str
        Reference allele.
    alt : list, str
        Alternate alleles.
    qual : int
        Variant quality.
    filter : str
        Variant filter codes.
    info : str
        Variant INFO string.
    format : str
        Variant format codes and sample values.

    Returns
    -------
    line : str
        VCF record line.
    """
    fields = [chrom, pos, id, ref, alt, qual, filter, info, format]
    return "\t".join(vcfstr(f) for f in fields)
