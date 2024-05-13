import pathlib
import tempfile
import os
import sys
import pytest

from mchap.application.atomize import main


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
