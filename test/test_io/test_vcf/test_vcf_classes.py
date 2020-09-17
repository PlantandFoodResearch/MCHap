import tempfile
import shutil
import pysam

from haplokit.io import vcf


def test_vcf_io():
    # use pysam to test VCF output

    # define samples (and order) present in output
    samples = (
        'SAMPLE1',
        'SAMPLE2',
        'SAMPLE3',
    )

    # construct header
    contigs = (
        vcf.ContigHeader('CHR1', 60),
        vcf.ContigHeader('CHR2', 60),
        vcf.ContigHeader('CHR3', 60),
    )
    meta_fields=(
        vcf.headermeta.fileformat('v4.3'),
        vcf.headermeta.filedate(),
        vcf.headermeta.source(),
        vcf.headermeta.phasing('None'),
        vcf.headermeta.commandline('.'),
        vcf.headermeta.randomseed(42),
    )
    filters=(
        vcf.filters.SamplePassFilter(),
        vcf.filters.SampleKmerFilter(),
        vcf.filters.SampleDepthFilter(),
        vcf.filters.SampleReadCountFilter(),
        vcf.filters.SamplePhenotypeProbabilityFilter(),
    )
    info_fields=(
        vcf.infofields.AN,
        vcf.infofields.AC,
        vcf.infofields.NS,
        vcf.infofields.END,
    )
    format_fields=(
        vcf.formatfields.GT,
        vcf.formatfields.GQ,
        vcf.formatfields.DP,
    )

    header = vcf.VCFHeader(
        meta=meta_fields,
        contigs=contigs,
        filters=filters,
        info_fields=info_fields,
        format_fields=format_fields,
        samples=samples,
    )

    # info field data
    # only fields that are defined in the 
    # header will appear in the output
    info_data = {
        'NA': 'This should no appear in output VCF',
        'AN': 2,
        'AC': [2, 1],
        #'NS' should appear as a null value '.'
        'END': 6,
    }

    # sample/format field data
    # only samples/fields that are defined in the 
    # header will appear in the output
    sample1 = {
        'GT': vcf.Genotype((0,0,2,-1), phased=False),
        'GQ': 60,
        'DP': 50,
        'NA': 'This should no appear in output VCF',
    }
    sample3 = {
        'GT': vcf.Genotype((0,0,1,2), phased=True),
        'GQ': 60,
        #'DP' should appear as a null value '.'
    }
    sample4 = {
        'GT': vcf.Genotype((1,1,1,1), phased=False),
        'GQ': 60,
        'DP': 50,
    }
    format_data = {
        'SAMPLE1': sample1,
        #'SAMPLE2' will have all null values in output
        'SAMPLE3': sample3,
        'SAMPLE4': sample4, # should not appear in output
    }

    # single record in VCF
    record = vcf.VCFRecord(
        header=header,
        chrom='CHR1',
        pos=5,
        id='VAR1',
        ref='A',
        alt=('C','G'),
        qual=None,
        filter='PASS',
        info=info_data,
        format=format_data,
    )

    # VCF data
    expect = vcf.VCF(
        header=header,
        records=[record]
    )

    # create temp file and write VCF data
    dirpath = tempfile.mkdtemp()
    tmp = dirpath + '/tmp.vcf'
    with open(tmp, 'w') as f:
        f.write(str(expect))

    # use pysam to read and compare written data
    actual = pysam.VariantFile(tmp)

    contig_expect = {(c.id, c.length) for c in expect.header.contigs}
    contig_actual = {(k, v.length) for k, v in actual.header.contigs.items()}
    assert contig_expect == contig_actual

    filter_code_expect = [f.id for f in expect.header.filters]
    filter_code_actual = actual.header.filters.keys()
    assert filter_code_expect == filter_code_actual

    # check header lines match
    header_expect = set(str(expect.header).strip().split('\n'))
    header_actual = set(str(actual.header).strip().split('\n'))
    assert header_expect == header_actual

    # check record lines match
    for i, record in enumerate(actual.fetch()):
        record_expect = str(expect.records[i]).strip()
        record_actual = str(record).strip()
        assert record_expect == record_actual
        
        # check samples are correct
        assert samples == tuple(record.samples.keys())

        # check sample gnotypes as expected
        assert record_actual.endswith('GT:GQ:DP\t0/0/2/.:60:50\t.:.:.\t0|0|1|2:60:.')

    shutil.rmtree(dirpath)

