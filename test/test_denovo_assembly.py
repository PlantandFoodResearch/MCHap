import pathlib
from haplokit.application.denovo_assembly import program


def test_Program__io():
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
    ]

    prog = program.cli(command)

    assert prog.mcmc_steps == 500
    assert prog.mcmc_burn == 100

    expect_sample_bam = dict(zip(samples, BAMS))
    assert len(prog.sample_bam) == len(expect_sample_bam)
    for k, v in expect_sample_bam.items():
        assert prog.sample_bam[k] == v

    expect_sample_ploidy = {sample: 4 for sample in samples}
    assert len(prog.sample_ploidy) == len(expect_sample_ploidy)
    for k, v in expect_sample_ploidy.items():
        assert prog.sample_ploidy[k] == v


def test_Program__run():
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
    ]

    prog = program.cli(command)
    out = prog.run()

    # check header
    contigs_expect = [
        '##contig=<ID=CHR1,length=60>',
        '##contig=<ID=CHR2,length=60>',
        '##contig=<ID=CHR3,length=60>',
    ]
    contigs_actual = [str(i) for i in out.header.contigs]
    assert contigs_actual == contigs_expect

    columns_expect = ('CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT')
    columns_expect += samples
    columns_actual = out.header.columns()
    assert columns_expect == columns_actual

    # check lines
    assert len(out.records) == 3



