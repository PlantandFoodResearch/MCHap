import numpy as np
import pathlib
import pytest
import pysam

from mchap.io import loci
from mchap.io import bam


@pytest.mark.parametrize(
    "id",
    ["SM", "ID"],
)
def test_extract_sample_ids(id):
    path = pathlib.Path(__file__).parent.absolute()
    paths = [
        str(path / "data/simple.sample1.bam"),
        str(path / "data/simple.sample2.bam"),
    ]

    if id == "SM":
        expect = {
            "SAMPLE1": paths[0],
            "SAMPLE2": paths[1],
        }
    elif id == "ID":
        expect = {
            "RG1_SAMPLE1": paths[0],
            "RG2_SAMPLE1": paths[0],  # multiple read groups for sample 1
            "RG1_SAMPLE2": paths[1],
        }

    actual = bam.extract_sample_ids(
        paths,
        id=id,
    )

    assert len(expect) == len(actual)
    for k, v in expect.items():
        assert actual[k] == v


def test_extract_read_variants():
    sample = "SAMPLE1"
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path / "data/simple.sample1.bam")

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

    expect_chars = np.array(
        [
            ["A", "A", "-"],
            ["A", "A", "-"],
            ["C", "G", "-"],
            ["A", "G", "-"],
            ["A", "A", "A"],
            ["A", "A", "A"],
            ["C", "G", "C"],
            ["A", "G", "T"],
            ["-", "A", "A"],
            ["-", "A", "A"],
            ["-", "G", "C"],
            ["-", "G", "T"],
            ["-", "A", "A"],
            ["-", "A", "A"],
            ["-", "G", "C"],
            ["-", "G", "T"],
            ["-", "-", "A"],
            ["-", "-", "A"],
            ["-", "-", "C"],
            ["-", "-", "T"],
        ],
        dtype="<U1",
    )
    expect_quals = np.zeros(expect_chars.shape, dtype=np.int16)
    expect_quals[expect_chars != "-"] = 50

    with pysam.AlignmentFile(path) as alignment_file:
        actual = bam.extract_read_variants(
            locus,
            alignment_file,
            samples=sample,
            id="SM",
        )
    assert sample in actual
    np.testing.assert_array_equal(actual[sample][0], expect_chars)
    np.testing.assert_array_equal(actual[sample][1], expect_quals)


def test_extract_read_variants__raise_on_ref():
    sample = "SAMPLE1"
    path = pathlib.Path(__file__).parent.absolute()
    path = str(path / "data/simple.sample1.bam")

    variants = (
        loci.SNP("CHR1", 6, 7, ".", alleles=("A", "C")),
        loci.SNP("CHR1", 15, 16, ".", alleles=("T", "G")),
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

    with pytest.raises(
        ValueError,
        match="Reference allele of variant 'T' does not match alignment reference allele 'A' at position 'CHR1:16' in target 'CHR1_05_25' in",
    ):
        with pysam.AlignmentFile(path) as alignment_file:
            bam.extract_read_variants(
                locus,
                alignment_file,
                samples=sample,
                id="SM",
            )


def test_encode_read_alleles():
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

    chars = np.array(
        [
            ["A", "A", "-"],
            ["A", "A", "-"],
            ["C", "G", "-"],
            ["A", "G", "-"],
            ["A", "A", "A"],
            ["A", "A", "A"],
            ["C", "G", "C"],
            ["A", "G", "T"],
            ["-", "A", "A"],
            ["-", "A", "A"],
            ["-", "G", "C"],
            ["-", "G", "T"],
            ["-", "A", "A"],
            ["-", "A", "A"],
            ["-", "G", "C"],
            ["-", "G", "T"],
            ["-", "-", "A"],
            ["-", "-", "A"],
            ["-", "-", "C"],
            ["-", "-", "T"],
        ],
        dtype="<U1",
    )

    expect = np.array(
        [
            [0, 0, -1],
            [0, 0, -1],
            [1, 1, -1],
            [0, 1, -1],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [0, 1, 2],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 1, 1],
            [-1, 1, 2],
            [-1, 0, 0],
            [-1, 0, 0],
            [-1, 1, 1],
            [-1, 1, 2],
            [-1, -1, 0],
            [-1, -1, 0],
            [-1, -1, 1],
            [-1, -1, 2],
        ],
        dtype=np.int8,
    )

    actual = bam.encode_read_alleles(locus, chars)
    np.testing.assert_array_equal(expect, actual)


def test_encode_read_distributions():
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

    alleles = np.array(
        [[0, 0, -1], [0, 0, -1], [1, 1, -1], [0, 1, -1], [-1, -1, 2]], dtype=np.int8
    )

    quals = np.zeros(alleles.shape, dtype=np.int16)
    quals[alleles >= 0] = 20
    error_rate = 0.001

    prob = 0.98901  # prob of correct call
    bi_call = prob
    bi_alt = (1 - prob) / 3
    tri_call = prob
    tri_alt = (1 - prob) / 3

    expect = np.array(
        [
            [[bi_call, bi_alt, 0.0], [bi_call, bi_alt, 0.0], [np.nan, np.nan, np.nan]],
            [[bi_call, bi_alt, 0.0], [bi_call, bi_alt, 0.0], [np.nan, np.nan, np.nan]],
            [[bi_alt, bi_call, 0.0], [bi_alt, bi_call, 0.0], [np.nan, np.nan, np.nan]],
            [[bi_call, bi_alt, 0.0], [bi_alt, bi_call, 0.0], [np.nan, np.nan, np.nan]],
            [
                [np.nan, np.nan, 0.0],
                [np.nan, np.nan, 0.0],
                [tri_alt, tri_alt, tri_call],
            ],
        ]
    )

    actual = bam.encode_read_distributions(
        locus,
        alleles,
        quals,
        error_rate=error_rate,
    )
    np.testing.assert_array_almost_equal(expect, actual)


def test_encode_read_distributions__zero_reads():
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

    n_read = 0
    n_pos = len(variants)
    max_allele = np.max(locus.count_alleles())

    calls = np.empty((n_read, n_pos), dtype=np.int8)
    quals = np.empty((n_read, n_pos), dtype=np.int16)

    expect = np.empty((n_read, n_pos, max_allele), dtype=float)
    actual = bam.encode_read_distributions(locus, calls, quals)
    print(expect.shape, actual.shape)
    np.testing.assert_array_equal(expect, actual)


def test_encode_read_distributions__zero_snps():
    variants = ()

    locus = loci.Locus(
        contig="CHR1",
        start=5,
        stop=25,
        name="CHR1_05_25",
        sequence="A" * 20,
        variants=variants,
    )

    n_read = 10
    n_pos = len(variants)
    max_allele = int(np.max(locus.count_alleles(), initial=0))

    calls = np.empty((n_read, n_pos), dtype=np.int8)
    quals = np.empty((n_read, n_pos), dtype=np.int16)
    expect = np.empty((n_read, n_pos, max_allele), dtype=float)
    actual = bam.encode_read_distributions(locus, calls, quals)
    np.testing.assert_array_equal(expect, actual)


def test_encode_read_distributions__zero_reads_or_snps():
    variants = ()

    locus = loci.Locus(
        contig="CHR1",
        start=5,
        stop=25,
        name="CHR1_05_25",
        sequence="A" * 20,
        variants=variants,
    )

    n_read = 0
    n_pos = len(variants)
    max_allele = int(np.max(locus.count_alleles(), initial=0))

    calls = np.empty((n_read, n_pos), dtype=np.int8)
    quals = np.empty((n_read, n_pos), dtype=np.int16)
    expect = np.empty((n_read, n_pos, max_allele), dtype=float)
    actual = bam.encode_read_distributions(locus, calls, quals)
    np.testing.assert_array_equal(expect, actual)
