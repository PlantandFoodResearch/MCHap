import pathlib
import tempfile
import os
import sys
import pytest

import numpy as np

from mchap.application.find_snvs import (
    main,
    bases_to_indices,
    _count_alleles,
    _vcf_sort_alleles,
    _order_as_vcf_alleles,
    format_allele_counts,
    format_genotype_calls,
    format_floats,
    format_samples_columns,
)


@pytest.mark.parametrize(
    "bams, extra, output_vcf",
    [
        (
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            [],
            "simple.output.basis.vcf",
        ),
        (
            [
                "simple.sample1.broken.cram",
                "simple.sample2.broken.cram",
                "simple.sample3.broken.cram",
            ],
            [],
            "simple.output.basis.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            [],
            "simple.output.basis.mixed_depth.vcf",  # NOTE: ADMF identical to "simple.output.basis.vcf"
        ),
        (
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            ["--minaf", "0.3"],
            "simple.output.basis.minaf0.3.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            ["--minad", "2"],
            "simple.output.basis.minad2.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            ["--minaf", "0.0", "--minad", "0"],
            "simple.output.basis.minaf0.minad0.vcf",
        ),
    ],
)
def test_main(bams, extra, output_vcf):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed")
    REFERENCE = str(path / "simple.fasta")
    BAMS = [str(path / bam) for bam in bams]

    command = (
        ["mchap", "find-snvs", "--targets", BED, "--reference", REFERENCE, "--bam"]
        + BAMS
        + extra
    )

    # capture stdout in file
    _, out_filename = tempfile.mkstemp()
    stdout = sys.stdout
    sys.stdout = open(out_filename, "w")
    main(command)
    sys.stdout.close()

    # replace stdout
    sys.stdout = stdout

    # compare output to expected
    with open(out_filename, "r") as f:
        actual = f.readlines()
    with open(str(path / output_vcf), "r") as f:
        expected = f.readlines()

    assert len(actual) == len(expected)

    for act, exp in zip(actual, expected):
        # file paths will make full line differ
        if act.startswith("##commandline"):
            assert exp.startswith("##commandline")
        elif act.startswith("##reference"):
            assert exp.startswith("##reference")
        elif act.startswith("##fileDate"):
            # new date should be greater than test vcf date
            assert exp.startswith("##fileDate")
            assert act > exp
        else:
            assert act == exp

    # cleanup
    os.remove(out_filename)


@pytest.mark.parametrize(
    "bases, expect",
    [
        (
            ["A", "C", "G", "T", "N", "a", "c", "g", "t", "n"],
            [0, 1, 2, 3, -1, 0, 1, 2, 3, -1],
        ),
        ([["A", "T"], ["C", "G"]], [[0, 3], [1, 2]]),
    ],
)
def test_bases_to_indices(bases, expect):
    actual = bases_to_indices(bases)
    expect = np.array(expect)
    np.testing.assert_array_equal(expect, actual)


@pytest.mark.parametrize(
    "alleles, counts",
    [
        ([1, 0, 1, 2, 1, 2], [1, 3, 2, 0]),
        ([1, -1, 3], [0, 1, 0, 1]),
    ],
)
def test__count_alleles(alleles, counts):
    actual = np.zeros(4, int)
    _count_alleles(actual, np.array(alleles))
    np.testing.assert_array_equal(counts, actual)


def test__vcf_sort_alleles():
    freqs = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.3, 0.5, 0.2],
            [0.0, 0.6, 0.1, 0.3],
        ]
    )
    ref_idx = np.array(
        [
            0,
            0,
            2,
        ]
    )
    excpect = np.array(
        [
            [0, 3, 2, 1],  # due to implementation but it doesn't matter
            [0, 2, 1, 3],
            [2, 1, 3, 0],
        ]
    )
    actual = _vcf_sort_alleles(freqs, ref_idx)
    np.testing.assert_array_equal(excpect, actual)


def test__order_as_vcf_alleles():
    order = np.array(
        [
            [0, 3, 2, 1],
            [0, 2, 1, 3],
            [2, 1, 3, 0],
        ]
    )
    # reference must be kept
    keep = np.array(
        [
            [True, False, False, False],
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    expect_ref = np.array(["A", "A", "G"])
    expect_alt = np.array(["", "G,C,T", "C,T"])
    actual_ref, actual_alt = _order_as_vcf_alleles(order, keep)
    np.testing.assert_array_equal(expect_ref, actual_ref)
    np.testing.assert_array_equal(expect_alt, actual_alt)


def test_format_allele_counts():
    counts = np.array(
        [
            [[0, 2, 3, 0], [0, 1, 3, 0]],
            [[4, 3, 1, 1], [3, 0, 7, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    keep = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    expect = np.array(
        [
            ["0,2,3", "0,1,3"],
            ["4,3,1", "3,0,7"],
            ["0,0", "0,0"],
        ]
    )
    actual = format_allele_counts(counts, keep)
    np.testing.assert_array_equal(expect, actual)


def test_format_floats():
    floats = np.array([0.123456, np.nan, 0.0, 1.0, 2.3, 10.0])
    expect = np.array(["0.123", ".", "0", "1", "2.3", "10"])
    actual = format_floats(floats)
    np.testing.assert_array_equal(expect, actual)


def test_format_genotype_calls():
    genotype_calls = np.array(
        [
            [[1, 2, -2, -2], [1, 2, 2, -1]],
            [[0, 1, -2, -2], [0, 0, 2, 2]],
            [[-1, -1, -2, -2], [-1, -1, -1, -1]],
        ]
    )
    actual = format_genotype_calls(genotype_calls)
    expect = np.array(
        [
            ["1/2", "1/2/2/."],
            ["0/1", "0/0/2/2"],
            ["./.", "./././."],
        ]
    )
    np.testing.assert_array_equal(expect, actual)


def test_format_samples_columns():
    genotype_calls = np.array(
        [
            [[1, 2, -2, -2], [1, 2, 2, -1]],
            [[0, 1, -2, -2], [0, 0, 2, 2]],
            [[-1, -1, -2, -2], [-1, -1, -1, -1]],
        ]
    )
    genotype_probs = np.array(
        [
            [0.9999, 0.999],
            [1.0, 0.0000123],
            [0.6666666667, 0.55],
        ]
    )
    allele_depths = np.array(
        [
            [[0, 2, 3, 0], [0, 1, 3, 0]],
            [[4, 3, 1, 1], [3, 0, 7, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    allele_keep = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    actual = format_samples_columns(
        genotype_calls, genotype_probs, allele_depths, allele_keep
    )
    expect = np.array(
        [
            ["GT:GPM:AD", "1/2:1:0,2,3", "1/2/2/.:0.999:0,1,3"],
            ["GT:GPM:AD", "0/1:1:4,3,1", "0/0/2/2:0:3,0,7"],
            ["GT:GPM:AD", "./.:0.667:0,0", "./././.:0.55:0,0"],
        ]
    )
    np.testing.assert_array_equal(expect, actual)


def test_format_samples_columns__no_genotypes():
    genotype_calls = None
    genotype_probs = None
    allele_depths = np.array(
        [
            [[0, 2, 3, 0], [0, 1, 3, 0]],
            [[4, 3, 1, 1], [3, 0, 7, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        ]
    )
    allele_keep = np.array(
        [
            [True, True, True, False],
            [True, True, True, False],
            [True, True, False, False],
        ]
    )
    actual = format_samples_columns(
        genotype_calls, genotype_probs, allele_depths, allele_keep
    )
    expect = np.array(
        [
            ["GT:AD", ".:0,2,3", ".:0,1,3"],
            ["GT:AD", ".:4,3,1", ".:3,0,7"],
            ["GT:AD", ".:0,0", ".:0,0"],
        ]
    )
    np.testing.assert_array_equal(expect, actual)
