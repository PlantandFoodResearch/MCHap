import pathlib
import pytest
import numpy as np

from mchap.io import loci


def test_read_bed4__plain_vs_zipped():
    directory = pathlib.Path(__file__).parent.absolute()
    plain = str(directory / "data/simple.bed")
    zipped = str(directory / "data/simple.bed.gz")

    expect = [
        loci.Locus(
            contig="CHR1",
            start=5,
            stop=25,
            name="CHR1_05_25",
            sequence=None,
            variants=None,
        ),
        loci.Locus(
            contig="CHR1",
            start=30,
            stop=50,
            name="CHR1_30_50",
            sequence=None,
            variants=None,
        ),
        loci.Locus(
            contig="CHR2",
            start=10,
            stop=30,
            name="CHR2_10_30",
            sequence=None,
            variants=None,
        ),
        loci.Locus(
            contig="CHR3",
            start=20,
            stop=40,
            name="CHR3_20_40",
            sequence=None,
            variants=None,
        ),
    ]

    actual = list(loci.read_bed4(plain))
    assert actual == expect

    actual = list(loci.read_bed4(zipped))
    assert actual == expect


def test_read_bed4__region():
    directory = pathlib.Path(__file__).parent.absolute()
    zipped = str(directory / "data/simple.bed.gz")

    intervals = [
        loci.Locus(
            contig="CHR1",
            start=5,
            stop=25,
            name="CHR1_05_25",
            sequence=None,
            variants=None,
        ),
        loci.Locus(
            contig="CHR1",
            start=30,
            stop=50,
            name="CHR1_30_50",
            sequence=None,
            variants=None,
        ),
        loci.Locus(
            contig="CHR2",
            start=10,
            stop=30,
            name="CHR2_10_30",
            sequence=None,
            variants=None,
        ),
        loci.Locus(
            contig="CHR3",
            start=20,
            stop=40,
            name="CHR3_20_40",
            sequence=None,
            variants=None,
        ),
    ]

    region = "CHR1"
    expect = intervals[0:2]
    actual = list(loci.read_bed4(zipped, region=region))
    assert actual == expect

    region = "CHR1:0-25"
    expect = intervals[0:1]
    actual = list(loci.read_bed4(zipped, region=region))
    assert actual == expect


def test_Locus__from_region_string():
    locus = loci.Locus.from_region_string("chrom1:123-100123")
    assert locus.name is None
    assert locus.contig == "chrom1"
    assert locus.start == 123
    assert locus.stop == 100123
    locus = loci.Locus.from_region_string("zrgf32.2:105-1001221343", name="ident105")
    assert locus.name == "ident105"
    assert locus.contig == "zrgf32.2"
    assert locus.start == 105
    assert locus.stop == 1001221343


def test_Locus__set_sequence():
    path = pathlib.Path(__file__).parent.absolute()
    ref = str(path / "data/simple.fasta")

    locus = loci.Locus(
        contig="CHR1", start=5, stop=25, name="CHR1_05_25", sequence=None, variants=None
    )

    expect = "A" * 20
    locus = locus.set_sequence(ref)
    assert locus.sequence == expect


def test_Locus__set_sequence__raise_on_ref():
    path = pathlib.Path(__file__).parent.absolute()
    ref = str(path / "data/wrong.fasta")
    vcf = str(path / "data/simple.vcf.gz")

    locus = loci.Locus(
        contig="CHR1", start=5, stop=25, name="CHR1_05_25", sequence=None, variants=None
    )
    locus = locus.set_variants(vcf)
    with pytest.raises(
        ValueError,
        match="Reference allele of variant 'A' does not match reference sequence 'T' at 'CHR1:7' in target 'CHR1_05_25'",
    ):
        locus = locus.set_sequence(ref)


def test_Locus__set_variants__raise_on_ref():
    path = pathlib.Path(__file__).parent.absolute()
    ref = str(path / "data/wrong.fasta")
    vcf = str(path / "data/simple.vcf.gz")

    locus = loci.Locus(
        contig="CHR1", start=5, stop=25, name="CHR1_05_25", sequence=None, variants=None
    )
    locus = locus.set_sequence(ref)
    with pytest.raises(
        ValueError,
        match="Reference allele of variant 'A' does not match reference sequence 'T' at 'CHR1:7' in target 'CHR1_05_25'",
    ):
        locus = locus.set_variants(vcf)


def test_Locus__set_variants__raise_on_not_indexes():
    path = pathlib.Path(__file__).parent.absolute()
    vcf = str(path / "data/simple.vcf")
    locus = loci.Locus(
        contig="CHR1", start=5, stop=25, name="CHR1_05_25", sequence=None, variants=None
    )
    with pytest.raises(
        ValueError,
        match="Could not fetch variants from file '.*simple.vcf', the file is not indexed",
    ):
        locus = locus.set_variants(vcf)


def test_Locus__set_variants__raise_on_no_file():
    path = pathlib.Path(__file__).parent.absolute()
    vcf = str(path / "data/does_not_exist.vcf")
    locus = loci.Locus(
        contig="CHR1", start=5, stop=25, name="CHR1_05_25", sequence=None, variants=None
    )
    with pytest.raises(
        FileNotFoundError,
    ):
        locus = locus.set_variants(vcf)


def test_Locus__set_variants__empty():
    path = pathlib.Path(__file__).parent.absolute()
    vcf = str(path / "data/simple.vcf.gz")

    # no variants under locus in VCF
    locus = loci.Locus(
        contig="CHR1", start=30, stop=50, name="CHR1_30_50", sequence=None, variants=()
    )

    expect = locus
    actual = locus.set_variants(vcf)
    assert expect == actual


def test_Locus__set_variants__simple():
    path = pathlib.Path(__file__).parent.absolute()
    vcf = str(path / "data/simple.vcf.gz")

    locus = loci.Locus(
        contig="CHR1",
        start=5,
        stop=25,
        name="CHR1_05_25",
        sequence="A" * 20,
        variants=None,
    )

    variants = (
        loci.SNP("CHR1", 6, 7, ".", alleles=("A", "C")),
        loci.SNP("CHR1", 15, 16, ".", alleles=("A", "G")),
        loci.SNP("CHR1", 22, 23, ".", alleles=("A", "C", "T")),
    )
    expect = loci.Locus(
        contig="CHR1",
        start=5,
        stop=25,
        name="CHR1_05_25",
        sequence="A" * 20,
        variants=variants,
    )

    actual = locus.set_variants(vcf)
    assert expect == actual


def test_Locus__set_variants__duplicate():
    path = pathlib.Path(__file__).parent.absolute()
    vcf = str(path / "data/simple.vcf.gz")

    locus = loci.Locus(
        contig="CHR2",
        start=10,
        stop=30,
        name="CHR2_10_30",
        sequence="A" * 20,
        variants=None,
    )

    # vcf contains 2 records at pos 20 which should be merged
    variants = (
        loci.SNP("CHR2", 14, 15, ".", alleles=("A", "T")),
        loci.SNP("CHR2", 19, 20, ".", alleles=("A", "C", "G", "T")),
    )
    expect = loci.Locus(
        contig="CHR2",
        start=10,
        stop=30,
        name="CHR2_10_30",
        sequence="A" * 20,
        variants=variants,
    )

    actual = locus.set_variants(vcf)
    assert expect == actual


def test_Locus__attributes():

    variants = (
        loci.SNP("CHR2", 14, 15, ".", alleles=("A", "T")),
        loci.SNP("CHR2", 19, 20, ".", alleles=("A", "C", "G", "T")),
    )
    locus = loci.Locus(
        contig="CHR2",
        start=10,
        stop=30,
        name="CHR2_10_30",
        sequence="A" * 20,
        variants=variants,
    )

    assert locus.positions == [14, 19]
    assert locus.alleles == [("A", "T"), ("A", "C", "G", "T")]
    assert locus.range == range(10, 30)
    assert locus.count_alleles() == [2, 4]


def test_Locus__format_haplotypes():

    variants = (
        loci.SNP("CHR1", 6, 7, ".", alleles=("A", "C")),
        loci.SNP("CHR1", 15, 16, ".", alleles=("A", "G")),
        loci.SNP("CHR1", 22, 23, ".", alleles=("A", "C", "T")),
    )
    locus = loci.Locus(
        contig="CHR1",
        start=5,
        stop=25,
        name="CHR1_05_25",
        sequence="A" * 20,
        variants=variants,
    )

    haplotypes = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.int8,
    )

    expect = np.array(
        [
            "AAAAAAAAAAAAAAAAAAAA",
            "AAAAAAAAAAAAAAAAAAAA",
            "ACAAAAAAAAGAAAAAACAA",
            "AAAAAAAAAAGAAAAAATAA",
        ]
    )

    actual = locus.format_haplotypes(haplotypes)
    np.testing.assert_array_equal(expect, actual)


def test_Locus__format_variants():

    variants = (
        loci.SNP("CHR1", 6, 7, ".", alleles=("A", "C")),
        loci.SNP("CHR1", 15, 16, ".", alleles=("A", "G")),
        loci.SNP("CHR1", 22, 23, ".", alleles=("A", "C", "T")),
    )
    locus = loci.Locus(
        contig="CHR1",
        start=5,
        stop=25,
        name="CHR1_05_25",
        sequence="A" * 20,
        variants=variants,
    )

    haplotypes = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.int8,
    )

    expect = np.array(
        [
            ["A", "A", "A"],
            ["A", "A", "A"],
            ["C", "G", "C"],
            ["A", "G", "T"],
        ]
    )

    actual = locus.format_variants(haplotypes)
    np.testing.assert_array_equal(expect, actual)
