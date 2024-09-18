import pathlib
import tempfile
import os
import sys
import shutil
import pysam
import pytest
import numpy as np

from mchap.version import __version__
from mchap.io.vcf.headermeta import filedate, columns
from mchap.application.assemble import program, _genotype_as_alleles


def local_file_path(name):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / ("test_io/data/" + name)
    return str(path)


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
        "assemble",
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
    samples = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]

    bams = [
        str(path / "simple.sample1.deep.bam"),
        str(path / "simple.sample2.deep.bam"),
        str(path / "simple.sample3.deep.bam"),
    ]

    sample_bams = {
        "SAMPLE1": str(path / "simple.sample1.deep.bam"),
        "SAMPLE2": str(path / "simple.sample2.deep.bam"),
        "SAMPLE3": str(path / "simple.sample3.deep.bam"),
    }

    # write some files to use in io
    dirpath = tempfile.mkdtemp()

    tmp_bam_list = dirpath + "/bams.txt"
    with open(tmp_bam_list, "w") as f:
        f.write("\n".join(bams))

    tmp_sample_ploidy = dirpath + "/sample-ploidy.txt"
    with open(tmp_sample_ploidy, "w") as f:
        f.write("SAMPLE3\t2\n")
        f.write("SAMPLE1\t6\n")
        f.write("SAMPLE2\t4\n")

    tmp_sample_inbreeding = dirpath + "/sample-inbreeding.txt"
    with open(tmp_sample_inbreeding, "w") as f:
        f.write("SAMPLE3\t0.1\n")
        f.write("SAMPLE1\t0.2\n")
        f.write("SAMPLE2\t0.0\n")

    tmp_sample_mcmc_temperatures = dirpath + "/sample-mcmc-temperatures.txt"
    with open(tmp_sample_mcmc_temperatures, "w") as f:
        f.write("SAMPLE3\t0.8\t0.1\t1\t0.2\n")  # out of order
        f.write("SAMPLE1\t0.2\n")  # missing cold chain
        # SAMPLE2 uses default

    command = [
        "mchap",
        "assemble",
        "--bam",
        tmp_bam_list,
        "--ploidy",
        tmp_sample_ploidy,
        "--inbreeding",
        tmp_sample_inbreeding,
        "--mcmc-temperatures",
        tmp_sample_mcmc_temperatures,
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
    assert prog.sample_bams == {k: [(k, v)] for k, v in sample_bams.items()}

    for k, v in zip(samples, [6, 4, 2]):
        assert prog.sample_ploidy[k] == v

    for k, v in zip(samples, [0.2, 0.0, 0.1]):
        assert prog.sample_inbreeding[k] == v

    temps = [
        [0.2, 1.0],  # sample 1
        [1.0],  # sample 2
        [0.1, 0.2, 0.8, 1.0],  # sample 3
    ]
    for k, v in zip(samples, temps):
        assert prog.sample_mcmc_temperatures[k] == v


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
        "assemble",
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
        '##FILTER=<ID=NOA,Description="No observed alleles at locus">',
        '##FILTER=<ID=AF0,Description="All alleles have prior allele frequency of zero">',
    ]
    filters_actual = [line for line in header if line.startswith("##FILTER")]
    assert filters_actual == filters_expect

    samples_expect = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]
    columns_expect = columns(samples_expect)
    columns_actual = [line for line in header if line.startswith("#CHROM")][0]
    assert columns_actual == columns_expect


@pytest.mark.parametrize(
    "bams,cli_extra,output_vcf",
    [
        (
            [
                "simple.sample1.bam",
                "simple.sample2.bam",
                "simple.sample3.bam",
            ],
            [],
            "simple.output.assemble.vcf",
        ),
        (
            [
                "simple.sample1.broken.cram",
                "simple.sample2.broken.cram",
                "simple.sample3.broken.cram",
            ],
            [],
            "simple.output.assemble.vcf",  # identical results from bam/cram
        ),
        (
            [
                "simple.sample1.deep.bam",
                "simple.sample2.deep.bam",
                "simple.sample3.deep.bam",
            ],
            [],
            "simple.output.deep.assemble.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            ["--report", "SNVDP"],
            "simple.output.mixed_depth.assemble.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            ["--report", "AFP"],
            "simple.output.mixed_depth.assemble.frequencies.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            ["--report", "ACP"],
            "simple.output.mixed_depth.assemble.counts.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            ["--report", "AOP", "AOPSUM"],
            "simple.output.mixed_depth.assemble.occurrence.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.deep.bam", "simple.sample3.bam"],
            ["--sample-pool", "POOL", "--report", "AFP"],
            "simple.output.mixed_depth.assemble.pool.frequencies.vcf",
        ),
        (
            ["simple.sample1.bam", "simple.sample2.bam", "simple.sample3.bam"],
            [
                "--haplotype-posterior-threshold",
                "1.0",
                "--base-error-rate",
                "0.0",
                "--use-base-phred-scores",
            ],
            "simple.output.nullallele.assemble.vcf",
        ),
        (
            [
                "simple.sample1.deep.bam",
                "simple.sample2.deep.bam",
                "simple.sample3.deep.bam",
            ],
            [
                "--ploidy",
                local_file_path("simple.pools-ploidy"),
                "--sample-pool",
                local_file_path("simple.pools"),
            ],
            "simple.output.deep.assemble.pools.vcf",
        ),
    ],
)
@pytest.mark.parametrize("cache_threshold", [-1, 10])
@pytest.mark.parametrize("n_cores", [1, 2])
def test_Program__run_stdout(bams, cli_extra, output_vcf, cache_threshold, n_cores):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    BED = str(path / "simple.bed.gz")
    VCF = str(path / "simple.vcf.gz")
    REF = str(path / "simple.fasta")
    BAMS = [str(path / bam) for bam in bams]

    command = (
        [
            "mchap",
            "assemble",
            "--bam",
        ]
        + BAMS
        + [
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
            "--mcmc-llk-cache-threshold",
            str(cache_threshold),
            "--cores",
            str(n_cores),
        ]
        + cli_extra
    )

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
        # versions will differ
        elif act.startswith("##source=mchap"):
            assert exp.startswith("##source=mchap")
        elif act.startswith("##fileDate"):
            # new date should be greater than test vcf date
            assert exp.startswith("##fileDate")
            assert act > exp
        else:
            assert act == exp

    # cleanup
    os.remove(out_filename)


@pytest.mark.parametrize(
    "region,region_id",
    [
        ("CHR1:5-25", "CHR1_05_25"),
        ("CHR1:30-50", "CHR1_30_50"),
        ("CHR2:10-30", "CHR2_10_30"),
        ("CHR3:20-40", "CHR3_20_40"),
    ],
)
@pytest.mark.parametrize("cache_threshold", [-1, 10])
def test_Program__run_stdout__region(region, region_id, cache_threshold):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    output_vcf = "simple.output.mixed_depth.assemble.vcf"

    REF = str(path / "simple.fasta")
    VCF = str(path / "simple.vcf.gz")

    # match sample to bam
    sample_bam_pairs = [
        "SAMPLE1" + "\t" + str(path / "simple.sample1.bam"),
        "SAMPLE2" + "\t" + str(path / "simple.sample2.deep.bam"),
        "SAMPLE3" + "\t" + str(path / "simple.sample3.bam"),
    ]

    # create a tmp file with sample bam pairs
    dirpath = tempfile.mkdtemp()
    tmp_sample_bams = dirpath + "/sample-bams.txt"
    with open(tmp_sample_bams, "w") as f:
        f.write("\n".join(sample_bam_pairs))

    # first part of VCF record line to match to
    contig, interval = region.split(":")
    start, _ = interval.split("-")
    record_start = "{}\t{}".format(contig, int(start) + 1)

    command = [
        "mchap",
        "assemble",
        "--bam",
        tmp_sample_bams,
        "--ploidy",
        "4",
        "--region",
        region,
        "--region-id",
        region_id,
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
        "--mcmc-llk-cache-threshold",
        str(cache_threshold),
        "--report",
        "SNVDP",
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
    with open(str(path / output_vcf), "r") as f:
        expected = f.readlines()
    # filter vcf down to the header and single expected record
    expected = [
        line
        for line in expected
        if (line.startswith("#") or line.startswith(record_start))
    ]

    # assert a single non-header line in actual and same number of lines as expect
    assert len([line for line in actual if (not line.startswith("#"))]) == 1
    assert len(actual) == len(expected)

    for act, exp in zip(actual, expected):
        # file paths will make full line differ
        if act.startswith("##commandline"):
            assert exp.startswith("##commandline")
        # versions will differ
        elif act.startswith("##source=mchap"):
            assert exp.startswith("##source=mchap")
        elif act.startswith("##fileDate"):
            # new date should be greater than test vcf date
            assert exp.startswith("##fileDate")
            assert act > exp
        else:
            assert act == exp

    # cleanup
    shutil.rmtree(dirpath)
    os.remove(out_filename)


def test_Program__output_pysam():
    """Test that program output can be parsed by pysam and that the
    parsed output matches the initial output.
    """
    path = pathlib.Path(__file__).parent.absolute()
    path = path / "test_io/data"

    OUTFILE = str(path / "simple.output.deep.assemble.vcf")

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
    OUTFILE = str(path / "simple.output.deep.assemble.vcf")

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
    OUTFILE = str(path / "simple.output.deep.assemble.vcf")

    reference = pysam.FastaFile(REFFILE)
    with pysam.VariantFile(OUTFILE) as vcf:
        for variant in vcf:
            # fetch with tuple of values expects a zero-based start
            ref_allele = reference.fetch(variant.contig, variant.start, variant.stop)
            assert ref_allele == variant.ref
    reference.close()


def test_genotype_as_alleles():
    haplotypes = np.array(
        [
            [0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    genotype = np.array(
        [
            [0, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
            [0, 1, 1],
        ],
        dtype=np.int8,
    )
    haplotype_labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}

    expect = np.array([0, 2, 2, -1])
    actual = _genotype_as_alleles(genotype, haplotype_labels)
    np.testing.assert_array_equal(actual, expect)
