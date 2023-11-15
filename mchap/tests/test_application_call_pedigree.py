import pathlib
import tempfile
import os
import sys
import pytest

from mchap.application.call_pedigree import program, ExperimentalFeatureWarning


def full_path(file_name):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / ("test_io/data/" + file_name)
    return str(path)


@pytest.mark.parametrize(
    "input_vcf,bams,pedigree,cli_extra,output_vcf",
    [
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            [],
            "simple.output.mixed_depth.call-pedigree.p132.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            ["--gamete-error", "0.5"],
            "simple.output.mixed_depth.call-pedigree.p132.gamerror0.5.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            ["--gamete-ibd", "0.1"],
            "simple.output.mixed_depth.call-pedigree.p132.lambda0.1.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            ["--gamete-ploidy", full_path("simple.tau.132.txt")],
            "simple.output.mixed_depth.call-pedigree.p132.tau-mixed.vcf",
        ),
        (
            "simple.output.mixed_depth.assemble.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            ["--report", "AFPRIOR", "AFP", "AOP", "GL", "GP"],
            "simple.output.mixed_depth.call-pedigree.p132.reportall.vcf",
        ),
        (
            "mock.input.frequencies.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            ["--prior-frequencies", "AFP"],
            "simple.output.mixed_depth.call-pedigree.p132.prior.vcf",
        ),
        (
            "mock.input.frequencies.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            ["--prior-frequencies", "AFP", "--report", "AFP", "AFPRIOR"],
            "simple.output.mixed_depth.call-pedigree.p132.frequencies.prior.vcf",
        ),
        (
            "mock.input.frequencies.vcf",
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            "simple.pedigree.132.txt",
            [
                "--prior-frequencies",
                "AFP",
                "--report",
                "AFP",
                "AFPRIOR",
                "--filter-input-haplotypes",
                "AFP>=0.1",
            ],
            "simple.output.mixed_depth.call-pedigree.p132.frequencies.skiprare.vcf",
        ),
    ],
)
@pytest.mark.parametrize("n_cores", [1, 2])
def test_Program__run_stdout(input_vcf, bams, pedigree, cli_extra, output_vcf, n_cores):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    VCF = str(path / input_vcf)
    BAMS = [str(path / bam) for bam in bams]
    PEDIGREE = str(path / pedigree)

    command = (
        [
            "mchap",
            "call-pedigree",
            "--bam",
        ]
        + BAMS
        + [
            "--sample-parents",
            PEDIGREE,
            "--ploidy",
            "4",
            "--haplotypes",
            VCF,
            "--mcmc-steps",
            "500",
            "--mcmc-burn",
            "100",
            "--mcmc-seed",
            "11",
            "--cores",
            str(n_cores),
        ]
        + cli_extra
    )

    with pytest.warns(ExperimentalFeatureWarning):
        prog = program.cli(command)

    # capture stdout in file
    _, out_filename = tempfile.mkstemp()
    stdout = sys.stdout
    sys.stdout = open(out_filename, "w")
    prog.run_stdout()
    sys.stdout.close()

    # replace stdout
    sys.stdout = stdout

    # compare output to expected
    with open(out_filename, "r") as f:
        actual = f.readlines()
    with open(str(path / output_vcf), "r") as f:
        expected = f.readlines()

    assert len(actual) == len(expected)

    if n_cores > 1:
        # output may be in different order
        actual.sort()
        expected.sort()

    for act, exp in zip(actual, expected):
        # file paths will make full line differ
        if act.startswith("##commandline"):
            assert exp.startswith("##commandline")
        elif act.startswith("##fileDate"):
            # new date should be greater than test vcf date
            assert exp.startswith("##fileDate")
            assert act > exp
        else:
            assert act == exp

    # cleanup
    os.remove(out_filename)
