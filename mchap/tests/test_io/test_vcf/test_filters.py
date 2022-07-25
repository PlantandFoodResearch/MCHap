from mchap.io.vcf import filters


def test_PASS():
    expect = '##FILTER=<ID=PASS,Description="All filters passed">'
    assert str(filters.PASS) == expect
    assert filters.PASS.id == "PASS"


def test_NOA():
    expect = '##FILTER=<ID=NOA,Description="No observed alleles at locus">'
    assert str(filters.NOA) == expect
    assert filters.NOA.id == "NOA"
