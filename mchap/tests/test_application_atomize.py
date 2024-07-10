import pathlib
import tempfile
import os
import sys
import pytest
import numpy as np

from mchap.application.atomize import (
    main,
    format_snv_alleles,
    get_haplotype_snv_indices,
    format_allele_floats,
)


def test_format_snv_alleles():
    haplotype_snvs = np.array(
        [
            ["A", "C", "A", "C"],
            ["A", "C", "G", "C"],
            ["A", "T", "T", "G"],
        ]
    )
    refs, alts, n_alts = format_snv_alleles(haplotype_snvs)
    np.testing.assert_array_equal(refs, ["A", "C", "A", "C"])
    np.testing.assert_array_equal(alts, ["", "T", "G,T", "G"])
    np.testing.assert_array_equal(n_alts, [0, 1, 2, 1])


def test_get_haplotype_snv_indices():
    haplotype_snvs = np.array(
        [
            ["A", "C", "A", "C"],
            ["A", "C", "G", "C"],
            ["A", "T", "T", "G"],
        ]
    )
    actual = get_haplotype_snv_indices(haplotype_snvs)
    expect = [
        [0, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 2, 1],
    ]
    np.testing.assert_array_equal(actual, expect)


def test_format_allele_floats__R():
    n_alts = np.array([1, 1, 2, 3])
    floats = np.arange(60).reshape(5, 3, 4) / 15
    floats[1] = np.nan
    floats[3, 1] = [0.0, 1.0, np.nan, 1.1]
    actual = format_allele_floats(floats, n_alts)
    expect = np.array(
        [
            ["0,0.067", "0.267,0.333", "0.533,0.6"],
            [".,.", ".,.", ".,."],
            ["1.6,1.667,1.733", "1.867,1.933,2", "2.133,2.2,2.267"],
            ["2.4,2.467,2.533,2.6", "0,1,.,1.1", "2.933,3,3.067,3.133"],
        ]
    )
    np.testing.assert_array_equal(actual, expect)


def test_format_allele_floats__A():
    n_alts = np.array([1, 1, 2, 3])
    floats = np.arange(60).reshape(5, 3, 4) / 15
    floats[1] = np.nan
    floats[3, 1] = [0.0, 1.0, np.nan, 1.1]
    actual = format_allele_floats(floats[:, :, 1:], n_alts, length="A")
    expect = np.array(
        [
            ["0.067", "0.333", "0.6"],
            [".", ".", "."],
            ["1.667,1.733", "1.933,2", "2.2,2.267"],
            ["2.467,2.533,2.6", "1,.,1.1", "3,3.067,3.133"],
        ]
    )
    np.testing.assert_array_equal(actual, expect)


@pytest.mark.parametrize(
    "input_vcf, output_vcf",
    [
        (
            "simple.output.mixed_depth.assemble.vcf",
            "simple.output.mixed_depth.assemble.atomize.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.counts.vcf",
            "simple.output.mixed_depth.assemble.counts.atomize.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.frequencies.vcf",
            "simple.output.mixed_depth.assemble.frequencies.atomize.vcf",
        ),
    ],
)
def test_main(input_vcf, output_vcf):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    INPUT = str(path / input_vcf)
    OUTPUT = str(path / output_vcf)

    command = ["mchap", "atomize", INPUT]

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
    with open(str(OUTPUT), "r") as f:
        expected = f.readlines()

    assert len(actual) == len(expected)

    for act, exp in zip(actual, expected):
        # file paths will make full line differ
        if act.startswith("##commandline"):
            assert exp.startswith("##commandline")
        # versions will differ
        elif act.startswith("##source=mchap"):
            assert exp.startswith("##source=mchap")
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
