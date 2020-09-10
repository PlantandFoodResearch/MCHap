import pytest
import pathlib

from haplokit.io import loci


def test_read_bed4__plain_vs_zipped():
    directiory = pathlib.Path(__file__).parent.absolute()
    plain = str(directiory / 'data/simple.bed')
    zipped = str(directiory / 'data/simple.bed.gz')

    expect = [
        loci.Locus(
            contig='CHR1', 
            start=5, 
            stop=25, 
            name='CHR1_05_25', 
            sequence=None, 
            variants=None
        ),
        loci.Locus(
            contig='CHR1', 
            start=30, 
            stop=50, 
            name='CHR1_30_50', 
            sequence=None, 
            variants=None
        ),
        loci.Locus(
            contig='CHR2', 
            start=10, 
            stop=30, 
            name='CHR2_10_30', 
            sequence=None, 
            variants=None
        )
    ]

    actual = list(loci.read_bed4(plain))
    assert actual == expect

    actual = list(loci.read_bed4(zipped))
    assert actual == expect


def test_read_bed4__region():
    directiory = pathlib.Path(__file__).parent.absolute()
    zipped = str(directiory / 'data/simple.bed.gz')

    intervals = [
        loci.Locus(
            contig='CHR1', 
            start=5, 
            stop=25, 
            name='CHR1_05_25', 
            sequence=None, 
            variants=None
        ),
        loci.Locus(
            contig='CHR1', 
            start=30, 
            stop=50, 
            name='CHR1_30_50', 
            sequence=None, 
            variants=None
        ),
        loci.Locus(
            contig='CHR2', 
            start=10, 
            stop=30, 
            name='CHR2_10_30', 
            sequence=None, 
            variants=None
        )
    ]

    region = 'CHR1'
    expect = intervals[0:2]
    actual = list(loci.read_bed4(zipped, region=region))
    assert actual == expect

    region = 'CHR1:0-25'
    expect = intervals[0:1]
    actual = list(loci.read_bed4(zipped, region=region))
    assert actual == expect


def test_Locus__set_sequence():
    path = pathlib.Path(__file__).parent.absolute() 
    ref = str(path / 'data/simple.fasta')

    locus = loci.Locus(
        contig='CHR1', 
        start=5, 
        stop=25, 
        name='CHR1_05_25', 
        sequence=None, 
        variants=None
    )

    expect = 'A' * 20
    locus = locus.set_sequence(ref)
    assert locus.sequence == expect


def test_Locus__set_variants__empty():
    path = pathlib.Path(__file__).parent.absolute() 
    vcf = str(path / 'data/simple.vcf.gz')
    
    # no variants under locus in VCF
    locus = loci.Locus(
        contig='CHR1', 
        start=30, 
        stop=50, 
        name='CHR1_30_50', 
        sequence=None, 
        variants=()
    )

    expect = locus
    actual = locus.set_variants(vcf)
    assert expect == actual


def test_Locus__set_variants__simple():
    path = pathlib.Path(__file__).parent.absolute() 
    vcf = str(path / 'data/simple.vcf.gz')

    locus = loci.Locus(
        contig='CHR1', 
        start=5, 
        stop=25, 
        name='CHR1_05_25', 
        sequence='A' * 20, 
        variants=None
    )

    variants = (
        loci.SNP('CHR1', 6, 7, '.', alleles=('A', 'C')),
        loci.SNP('CHR1', 15, 16, '.', alleles=('A', 'G')),
        loci.SNP('CHR1', 22, 23, '.', alleles=('A', 'C', 'T')),
    )
    expect = loci.Locus(
        contig='CHR1', 
        start=5, 
        stop=25, 
        name='CHR1_05_25', 
        sequence='A' * 20, 
        variants=variants
    )

    actual = locus.set_variants(vcf)
    assert expect == actual


def test_Locus__set_variants__duplicate():
    path = pathlib.Path(__file__).parent.absolute() 
    vcf = str(path / 'data/simple.vcf.gz')

    locus = loci.Locus(
        contig='CHR2', 
        start=10, 
        stop=30, 
        name='CHR2_10_30', 
        sequence='A' * 20, 
        variants=None
    )

    # vcf contains 2 records at pos 20 which should be merged
    variants = (
        loci.SNP('CHR2', 14, 15, '.', alleles=('A', 'T')),
        loci.SNP('CHR2', 19, 20, '.', alleles=('A', 'C', 'G', 'T')),
    )
    expect = loci.Locus(
        contig='CHR2', 
        start=10, 
        stop=30, 
        name='CHR2_10_30', 
        sequence='A' * 20, 
        variants=variants
    )

    actual = locus.set_variants(vcf)
    assert expect == actual
