import pytest
import pathlib
import pysam
import numpy as np
from mchap.io.filter_alleles import parse_allele_filter, apply_allele_filter


@pytest.mark.parametrize(
    "string, expect",
    [
        ("AD>0", ("AD", np.greater, int(0))),
        ("AFP>0.0", ("AFP", np.greater, 0.0)),
        ("AFP>=0.0", ("AFP", np.greater_equal, 0.0)),
        ("AFP<0.0", ("AFP", np.less, 0.0)),
        ("AFP<=0.0", ("AFP", np.less_equal, 0.0)),
        ("AFP=0.0", ("AFP", np.equal, 0.0)),
        ("AFP==0.0", ("AFP", np.equal, 0.0)),
        ("AFP!=0.0", ("AFP", np.not_equal, 0.0)),
    ],
)
def test_parse_allele_filter(string, expect):
    actual = parse_allele_filter(string)
    assert actual == expect
    assert type(actual[-1]) == type(expect[-1])  # noqa: E721


@pytest.mark.parametrize("string", ["AD1", "AD-1", "AD~1", "AD>AFD", "", ">0.1"])
def test_parse_allele_filter__raise_on_invalid_string(string):
    with pytest.raises(ValueError, match=f"Invalid allele filter '{string}'"):
        parse_allele_filter(string)


@pytest.mark.parametrize("string, operator", [("AD<>1", "<>")])
def test_parse_allele_filter__raise_on_invalid_operator(string, operator):
    with pytest.raises(
        ValueError, match=f"Invalid operator in allele filter '{operator}'"
    ):
        parse_allele_filter(string)


@pytest.mark.parametrize(
    "field, func, value, expect",
    [
        ("AC", np.greater, 2, [True, True, False]),
        ("AC", np.greater_equal, 2, [True, True, True]),
        ("AFP", np.greater_equal, 0.3, [True, False, False]),
    ],
)
def test_apply_allele_filter(field, func, value, expect):
    directory = pathlib.Path(__file__).parent.absolute()
    path = str(directory / "data/simple.output.mixed_depth.assemble.frequencies.vcf")
    with pysam.VariantFile(path) as vcf:
        record = next(vcf.fetch())
        actual = apply_allele_filter(record, field, func, value)
    np.testing.assert_array_almost_equal(expect, actual)


@pytest.mark.parametrize(
    "field, func, value, message",
    [
        ("ACC", np.greater, 2, "Allele filter field not found in header 'ACC'"),
        ("DP", np.greater, 2, "Allele filter of field of invalid length '1'"),
        ("SNVPOS", np.greater, 2, "Allele filter of field of invalid length '.'"),
    ],
)
def test_apply_allele_filter__raise(field, func, value, message):
    directory = pathlib.Path(__file__).parent.absolute()
    path = str(directory / "data/simple.output.mixed_depth.assemble.frequencies.vcf")
    with pysam.VariantFile(path) as vcf:
        record = next(vcf.fetch())
        with pytest.raises(ValueError, match=message):
            apply_allele_filter(record, field, func, value)
