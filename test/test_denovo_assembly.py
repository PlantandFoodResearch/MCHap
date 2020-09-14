import pathlib

from haplokit.version import __version__
from haplokit.io.vcf.headermeta import filedate
from haplokit.application.denovo_assembly import program


def test_Program__cli():
    samples = ('SAMPLE1', 'SAMPLE2', 'SAMPLE3')

    path = pathlib.Path(__file__).parent.absolute()
    path = path / 'test_io/data'

    BED = str(path / 'simple.bed.gz')
    VCF = str(path / 'simple.vcf.gz')
    REF = str(path / 'simple.fasta')
    BAMS = [
        str(path / 'simple.sample1.deep.bam'),
        str(path / 'simple.sample2.deep.bam'),
        str(path / 'simple.sample3.deep.bam')
    ]

    command = [
        'haplokit-denovo',
        '--bam', BAMS[0], BAMS[1], BAMS[2],
        '--ploidy', '4',
        '--bed', BED,
        '--vcf', VCF,
        '--ref', REF,
        '--mcmc-steps', '500',
        '--mcmc-burn', '100',
        '--seed', '11',
        '--cores', '5',
    ]

    prog = program.cli(command)

    assert prog.mcmc_steps == 500
    assert prog.mcmc_burn == 100
    assert prog.random_seed == 11
    assert prog.n_cores == 5
    assert prog.cli_command == command

    expect_sample_bam = dict(zip(samples, BAMS))
    assert len(prog.sample_bam) == len(expect_sample_bam)
    for k, v in expect_sample_bam.items():
        assert prog.sample_bam[k] == v

    expect_sample_ploidy = {sample: 4 for sample in samples}
    assert len(prog.sample_ploidy) == len(expect_sample_ploidy)
    for k, v in expect_sample_ploidy.items():
        assert prog.sample_ploidy[k] == v


def test_Program__header():
    path = pathlib.Path(__file__).parent.absolute()
    path = path / 'test_io/data'

    BED = str(path / 'simple.bed.gz')
    VCF = str(path / 'simple.vcf.gz')
    REF = str(path / 'simple.fasta')
    BAMS = [
        str(path / 'simple.sample1.deep.bam'),
        str(path / 'simple.sample2.deep.bam'),
        str(path / 'simple.sample3.deep.bam')
    ]

    command = [
        'haplokit-denovo',
        '--bam', BAMS[0], BAMS[1], BAMS[2],
        '--ploidy', '4',
        '--bed', BED,
        '--vcf', VCF,
        '--ref', REF,
        '--mcmc-steps', '500',
        '--mcmc-burn', '100',
        '--seed', '11',
    ]

    prog = program.cli(command)
    header = prog.header()

    meta_expect = [
        '##fileformat=VCFv4.3',
        str(filedate()),
        '##source=Haplokit v{}'.format(__version__),
        '##phasing=None',
        '##commandline="{}"'.format(' '.join(command)),
        '##randomseed=11',
    ]
    meta_actual = [str(i) for i in header.meta]
    assert meta_actual == meta_expect

    contigs_expect = [
        '##contig=<ID=CHR1,length=60>',
        '##contig=<ID=CHR2,length=60>',
        '##contig=<ID=CHR3,length=60>',
    ]
    contigs_actual = [str(i) for i in header.contigs]
    assert contigs_actual == contigs_expect

    filters_expect = [
        r'##FILTER=<ID=k3<0.95,Description="Less than 95.0 % of samples read-variant 3-mers ">',
        r'##FILTER=<ID=dp<5.0,Description="Sample has mean read depth less than 5.0.">',
        r'##FILTER=<ID=rc<5.0,Description="Sample has read (pair) count of less than 5.0 in haplotype interval.">',
        r'##FILTER=<ID=pp<0.95,Description="Samples phenotype posterior probability < 0.95.">',
    ]
    filters_actual = [str(i) for i in header.filters]
    assert filters_actual == filters_expect


    samples_expect = ('SAMPLE1', 'SAMPLE2', 'SAMPLE3')
    samples_actual = header.samples
    assert samples_actual == samples_expect

    columns_expect = ('CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT')
    columns_expect += samples_expect
    columns_actual = header.columns()
    assert columns_expect == columns_actual

