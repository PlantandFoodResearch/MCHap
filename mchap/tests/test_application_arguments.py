import pathlib
import pytest

from mchap.application.arguments import parse_sample_pools


def local_file_path(name):
    path = pathlib.Path(__file__).parent.absolute()
    path = path / ("test_io/data/" + name)
    return str(path)


def test_parse_sample_pools__none():
    samples = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]
    sample_bams = {"SAMPLE1": "BAM1", "SAMPLE2": "BAM2", "SAMPLE3": "BAM3"}
    pools, pool_bams = parse_sample_pools(
        samples, sample_bams, sample_pool_argument=None
    )
    assert pools == samples
    assert pool_bams == {
        "SAMPLE1": [("SAMPLE1", "BAM1")],
        "SAMPLE2": [("SAMPLE2", "BAM2")],
        "SAMPLE3": [("SAMPLE3", "BAM3")],
    }


def test_parse_sample_pools__single():
    samples = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]
    sample_bams = {"SAMPLE1": "BAM1", "SAMPLE2": "BAM2", "SAMPLE3": "BAM3"}
    pools, pool_bams = parse_sample_pools(
        samples, sample_bams, sample_pool_argument="POOL"
    )
    assert pools == ["POOL"]
    assert pool_bams == {
        "POOL": [("SAMPLE1", "BAM1"), ("SAMPLE2", "BAM2"), ("SAMPLE3", "BAM3")]
    }


def test_parse_sample_pools__file():
    samples = ["SAMPLE1", "SAMPLE2", "SAMPLE3"]
    sample_bams = {"SAMPLE1": "BAM1", "SAMPLE2": "BAM2", "SAMPLE3": "BAM3"}
    pools, pool_bams = parse_sample_pools(
        samples, sample_bams, sample_pool_argument=local_file_path("simple.pools")
    )
    assert pools == ["POOL1", "POOL2", "POOL3", "POOL13", "POOL123"]
    assert pool_bams == {
        "POOL1": [("SAMPLE1", "BAM1")],
        "POOL2": [("SAMPLE2", "BAM2")],
        "POOL3": [("SAMPLE3", "BAM3")],
        "POOL13": [("SAMPLE1", "BAM1"), ("SAMPLE3", "BAM3")],
        "POOL123": [("SAMPLE1", "BAM1"), ("SAMPLE2", "BAM2"), ("SAMPLE3", "BAM3")],
    }


def test_parse_sample_pools__raise_on_missing_sample():
    samples = ["SAMPLE1", "SAMPLE2", "SAMPLE3", "SAMPLE4"]
    sample_bams = {"SAMPLE1": "BAM1", "SAMPLE2": "BAM2", "SAMPLE3": "BAM3"}
    with pytest.raises(
        ValueError,
        match="The following samples have not been assigned to a pool: {'SAMPLE4'}",
    ):
        parse_sample_pools(
            samples, sample_bams, sample_pool_argument=local_file_path("simple.pools")
        )


def test_parse_sample_pools__raise_on_unknown_sample():
    samples = ["SAMPLE1", "SAMPLE2"]
    sample_bams = {"SAMPLE1": "BAM1", "SAMPLE2": "BAM2", "SAMPLE3": "BAM3"}
    with pytest.raises(
        ValueError,
        match="The following names in the sample-pool file do not match a known sample : {'SAMPLE3'}",
    ):
        parse_sample_pools(
            samples, sample_bams, sample_pool_argument=local_file_path("simple.pools")
        )
