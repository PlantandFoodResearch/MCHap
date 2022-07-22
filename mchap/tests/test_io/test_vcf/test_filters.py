from mchap.io.vcf import filters


def test_PASS():
    expect = '##FILTER=<ID=PASS,Description="All filters passed">'
    assert str(filters.PASS) == expect
    assert filters.PASS.id == "PASS"


def test_NAA():
    expect = '##FILTER=<ID=NAA,Description="No alleles assembled with probability greater than threshold">'
    assert str(filters.NAA) == expect
    assert filters.NAA.id == "NAA"
