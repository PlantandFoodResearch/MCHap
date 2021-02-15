import pathlib
import tempfile
import os
import sys
import shutil
import pysam
import pytest

from mchap.version import __version__
from mchap.io.vcf.headermeta import filedate, columns
from mchap.application.assemble import program


def test_Program__cli():
    samples = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]

    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")
    BAMS = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    command = [
        "mchap",
        "denovo",
        "--bam",
        BAMS[0],
        BAMS[1],
        BAMS[2],
        "--ploidy",
        "4",
        "--targets",
        BED,
        "--variants",
        VCF,
        "--reference",
        REF,
        "--mcmc-steps",
        "500",
        "--mcmc-burn",
        "100",
        "--mcmc-seed",
        "11",
        "--cores",
        "5",
    ]

    prog = program.cli(command)

    assert prog.mcmc_steps == 500
    assert prog.mcmc_burn == 100
    assert prog.random_seed == 11
    assert prog.n_cores == 5
    assert prog.cli_command == command

    assert prog.samples == samples

    expect_sample_ploidy = {sample: 4 for sample in samples}
    assert len(prog.sample_ploidy) == len(expect_sample_ploidy)
    for k, v in expect_sample_ploidy.items():
        assert prog.sample_ploidy[k] == v


def test_Program__cli_lists():
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")

    # partial overlap with bam samples
    samples = ["SAMPLE3", "SAMPLE4", "SAMPLE1"]

    bams = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    # write some files to use in io
    dirpath = tempfile.mkdtemp()

    tmp_bam_list = dirpath + "/bams.txt"
    with open(tmp_bam_list, "w") as f:
        f.write("\n".join(bams))

    tmp_sample_list = dirpath + "/samples.txt"
    with open(tmp_sample_list, "w") as f:
        f.write("\n".join(samples))

    tmp_sample_ploidy = dirpath + "/sample-ploidy.txt"
    with open(tmp_sample_ploidy, "w") as f:
        f.write("SAMPLE3\t2\n")
        f.write("SAMPLE1\t6\n")
        # SAMPLE4 uses default

    tmp_sample_inbreeding = dirpath + "/sample-inbreeding.txt"
    with open(tmp_sample_inbreeding, "w") as f:
        f.write("SAMPLE3\t0.1\n")
        f.write("SAMPLE1\t0.2\n")
        # SAMPLE4 uses default

    command = [
        "mchap",
        "denovo",
        "--bam-list",
        tmp_bam_list,
        "--sample-list",
        tmp_sample_list,
        "--sample-ploidy",
        tmp_sample_ploidy,
        "--ploidy",
        "4",
        "--sample-inbreeding",
        tmp_sample_inbreeding,
        "--targets",
        BED,
        "--variants",
        VCF,
        "--reference",
        REF,
        "--mcmc-steps",
        "500",
        "--mcmc-burn",
        "100",
        "--mcmc-seed",
        "11",
        "--cores",
        "5",
    ]

    prog = program.cli(command)

    # clean up
    shutil.rmtree(dirpath)

    assert prog.mcmc_steps == 500
    assert prog.mcmc_burn == 100
    assert prog.random_seed == 11
    assert prog.n_cores == 5
    assert prog.cli_command == command

    assert prog.samples == samples
    assert prog.bams == bams

    for k, v in zip(samples, [2, 4, 6]):
        assert prog.sample_ploidy[k] == v

    for k, v in zip(samples, [0.1, 0.0, 0.2]):
        assert prog.sample_inbreeding[k] == v


def test_Program__header():
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")
    BAMS = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    command = [
        "mchap",
        "denovo",
        "--bam",
        BAMS[0],
        BAMS[1],
        BAMS[2],
        "--ploidy",
        "4",
        "--targets",
        BED,
        "--variants",
        VCF,
        "--reference",
        REF,
        "--mcmc-steps",
        "500",
        "--mcmc-burn",
        "100",
        "--mcmc-seed",
        "11",
    ]

    prog = program.cli(command)
    header = prog.header()

    # meta lines should be at the top
    meta_expect = [
        "##fileformat=VCFv4.3",
        str(filedate()),
        "##source=mchap v{}".format(__version__),
        "##phasing=None",
        '##commandline="{}"'.format(" ".join(command)),
        "##randomseed=11",
    ]
    meta_actual = header[0:6]
    assert meta_actual == meta_expect

    contigs_expect = [
        "##contig=<ID=CHR1,length=60>",
        "##contig=<ID=CHR2,length=60>",
        "##contig=<ID=CHR3,length=60>",
    ]
    contigs_actual = [line for line in header if line.startswith("##contig")]
    assert contigs_actual == contigs_expect

    filters_expect = [
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FILTER=<ID=3m90,Description="Less than 90.0 percent of read-variant 3-mers represented in haplotypes">',
        '##FILTER=<ID=dp5,Description="Sample has mean read depth less than 5.0">',
        '##FILTER=<ID=rc5,Description="Sample has read (pair) count of less than 5.0">',
        '##FILTER=<ID=pp95,Description="Samples phenotype posterior probability less than 0.95">',
        '##FILTER=<ID=mci60,Description="Replicate Markov chains found incongruent phenotypes with posterior probability greater than 0.6">',
        '##FILTER=<ID=cnv60,Description="Combined chains found more haplotypes than ploidy with posterior probability greater than 0.6">',
    ]
    filters_actual = [line for line in header if line.startswith("##FILTER")]
    assert filters_actual == filters_expect

    samples_expect = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]
    columns_expect = columns(samples_expect)
    columns_actual = [line for line in header if line.startswith("#CHROM")][0]
    assert columns_actual == columns_expect


def test_Program__run():
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")
    BAMS = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    command = [
        "mchap",
        "denovo",
        "--bam",
        BAMS[0],
        BAMS[1],
        BAMS[2],
        "--ploidy",
        "4",
        "--targets",
        BED,
        "--variants",
        VCF,
        "--reference",
        REF,
        "--mcmc-steps",
        "500",
        "--mcmc-burn",
        "100",
        "--mcmc-seed",
        "11",
    ]

    prog = program.cli(command)
    result = prog.run()

    # compare to expected VCF
    with open(str(path / "simple.output.deep.vcf"), "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("##commandline"):
                # file paths will differ
                pass
            elif line.startswith("##fileDate"):
                # new date should be greater than test vcf date
                assert result[i] > line
            else:
                assert result[i] == line


def test_Program__run__no_base_phreds():
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")
    BAMS = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    command = [
        "mchap",
        "denovo",
        "--bam",
        BAMS[0],
        BAMS[1],
        BAMS[2],
        "--ploidy",
        "4",
        "--targets",
        BED,
        "--variants",
        VCF,
        "--reference",
        REF,
        "--base-error-rate",
        "0.001",
        "--ignore-base-phred-scores",
        "--mcmc-steps",
        "500",
        "--mcmc-burn",
        "100",
        "--mcmc-seed",
        "11",
    ]

    prog = program.cli(command)
    result = prog.run()

    # compare to expected VCF
    with open(str(path / "simple.output.deep.vcf"), "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line.startswith("##commandline"):
                # file paths will differ
                pass
            elif line.startswith("##fileDate"):
                # new date should be greater than test vcf date
                assert result[i] > line
            else:
                assert result[i] == line


@pytest.mark.parametrize("n_cores", [1, 2])
def test_Program__run_stdout(n_cores):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")
    BAMS = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    command = [
        "mchap",
        "denovo",
        "--bam",
        BAMS[0],
        BAMS[1],
        BAMS[2],
        "--ploidy",
        "4",
        "--targets",
        BED,
        "--variants",
        VCF,
        "--reference",
        REF,
        "--mcmc-steps",
        "500",
        "--mcmc-burn",
        "100",
        "--mcmc-seed",
        "11",
        "--cores",
        str(n_cores),
    ]

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
    with open(str(path / "simple.output.deep.vcf"), "r") as f:
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


def test_Program__output_pysam():
    """Test that program output can be parsed by pysam and that the
    parsed output matches the initial output.
    """
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    OUTFILE = str(path / "simple.output.deep.vcf")

    with open(OUTFILE, "r") as f:
        expect = set(line.strip() for line in f.readlines())
        expect -= {""}

    with pysam.VariantFile(OUTFILE) as f:
        header = set(str(f.header).split("\n"))
        records = set(str(r).strip() for r in f)
        actual = header | records
        actual -= {""}

    assert expect == actual


def test_Program__output_bed_positions():
    """Tests that 1-based VCF intervals match the initial 0-based BED intervals.

    Note that with pysam variant objects the `.pos` attribute returns the 1-based
    position from the VCF and the `.start` attribute returns the adjusted 0-based
    position.
    Pysam `.fetch` methods expect 0-based integer arguments.
    """
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BEDFILE = str(path / "simple.bed")
    OUTFILE = str(path / "simple.output.deep.vcf")

    # map of named intervals from bed4 file
    with open(BEDFILE) as bed:
        intervals = bed.readlines()
    intervals = [line.strip().split("\t") for line in intervals]
    intervals = {
        name: (contig, int(start), int(stop)) for contig, start, stop, name in intervals
    }

    with pysam.VariantFile(OUTFILE) as vcf:
        for variant in vcf:
            # use 0-based start to match 0-based bed file
            name = variant.id
            interval = (variant.contig, variant.start, variant.stop)
            assert intervals[name] == interval


def test_Program__output_reference_positions():
    """Tests that VCF reference alleles match the reference genome.

    Note that with pysam variant objects the `.pos` attribute returns the 1-based
    position from the VCF and the `.start` attribute returns the adjusted 0-based
    position.
    Pysam `.fetch` methods expect 0-based integer arguments.
    """
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    REFFILE = str(path / "simple.fasta")
    OUTFILE = str(path / "simple.output.deep.vcf")

    reference = pysam.FastaFile(REFFILE)
    with pysam.VariantFile(OUTFILE) as vcf:
        for variant in vcf:
            # fetch with tuple of values expects a zero-based start
            ref_allele = reference.fetch(variant.contig, variant.start, variant.stop)
            assert ref_allele == variant.ref
    reference.close()
