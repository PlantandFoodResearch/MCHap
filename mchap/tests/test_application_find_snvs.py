import pathlib
import tempfile
import os
import sys
import pytest

from mchap.application.find_snvs import main


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
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            ["--maf", "0.2"],
            "simple.output.basis.maf0.2.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            ["--maf", "0.0"],
            "simple.output.basis.maf0.vcf",
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
