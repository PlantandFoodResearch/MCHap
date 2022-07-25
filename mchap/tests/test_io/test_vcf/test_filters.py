from mchap.io.vcf import filters


def test_PASS():
    expect = '##FILTER=<ID=PASS,Description="All filters passed">'
    assert str(filters.PASS) == expect
    assert filters.PASS.id == "PASS"


def test_NOA():
    expect = '##FILTER=<ID=NOA,Description="No observed alleles at locus">'
    assert str(filters.NOA) == expect
    assert filters.NOA.id == "NOA"


def test_AF0():
    expect = '##FILTER=<ID=AF0,Description="All alleles have prior allele frequency of zero">'
    assert str(filters.AF0) == expect
    assert filters.AF0.id == "AF0"
