import numpy as np
import pytest

from mchap.io.vcf import filters
from mchap.assemble.classes import GenotypeMultiTrace


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(
            filters.FilterCall("pp95", failed=False, applied=True), "PASS", id="pass"
        ),
        pytest.param(
            filters.FilterCall("pp95", failed=True, applied=True), "pp95", id="fail"
        ),
        pytest.param(
            filters.FilterCall("pp95", failed=False, applied=False),
            ".",
            id="not-applied",
        ),
    ],
)
def test_FilterCall__str(obj, expect):
    actual = str(obj)
    assert actual == expect


def test_FilterCallSet__str():
    calls = filters.FilterCallSet(
        (
            filters.FilterCall("pp95", failed=True, applied=True),
            filters.FilterCall("3m95", failed=True, applied=True),
            filters.FilterCall("dp5", failed=False, applied=True),
            filters.FilterCall("nan", failed=False, applied=False),
        )
    )
    expect = "pp95,3m95"
    actual = str(calls)
    assert expect == actual


def test_SamplePassFilter():
    obj = filters.SamplePassFilter()
    expect = '##FILTER=<ID=PASS,Description="All filters passed">'
    assert str(obj) == expect
    assert obj.id == "PASS"


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(filters.SampleKmerFilter(k=3, threshold=0.75), "PASS", id="pass"),
        pytest.param(filters.SampleKmerFilter(k=3, threshold=0.99), "3m99", id="fail"),
        pytest.param(
            filters.SampleKmerFilter(k=5, threshold=0.95), ".", id="not-applied"
        ),
    ],
)
def test_SampleKmerFilter(obj, expect):
    genotype = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 1, 0, 1], [1, 1, 1, 1]])
    read_variants = np.array(
        [
            [-1, -1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 1, 0, -1],
            [1, -1, 1, 1],
            [0, 1, 1, -1],  # 011- is not in haplotypes
        ]
    )
    # apply filter and get FT code
    actual = str(obj(read_variants, genotype))
    assert actual == expect


@pytest.mark.parametrize(
    "obj,depth,expect",
    [
        pytest.param(
            filters.SampleDepthFilter(threshold=5),
            5.5,
            "PASS",
            id="pass",
        ),
        pytest.param(
            filters.SampleDepthFilter(threshold=6),
            5.5,
            "dp6",
            id="fail",
        ),
        pytest.param(
            filters.SampleDepthFilter(threshold=6), np.nan, ".", id="not-applied"
        ),
    ],
)
def test_SampleDepthFilter(obj, depth, expect):
    actual = str(obj(depth))
    assert actual == expect


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(filters.SampleReadCountFilter(threshold=5), "PASS", id="pass"),
        pytest.param(filters.SampleReadCountFilter(threshold=6), "rc6", id="fail"),
    ],
)
def test_SampleReadCountFilter(obj, expect):
    count = 5
    actual = str(obj(count))
    assert actual == expect


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(
            filters.SamplePhenotypeProbabilityFilter(threshold=0.95), "PASS", id="pass"
        ),
        pytest.param(
            filters.SamplePhenotypeProbabilityFilter(threshold=0.99), "pp99", id="fail"
        ),
    ],
)
def test_SamplePhenotypeProbabilityFilter(obj, expect):
    prob = 0.9773
    actual = str(obj(prob))
    assert actual == expect


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(
            filters.SampleChainPhenotypeIncongruenceFilter(threshold=0.8),
            "PASS",
            id="pass",
        ),
        pytest.param(
            filters.SampleChainPhenotypeIncongruenceFilter(threshold=0.6),
            "mci60",
            id="fail",
        ),
    ],
)
def test_SampleChainPhenotypeIncongruenceFilter(obj, expect):
    haplotypes = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    g0 = haplotypes[[0, 0, 1, 2]]  # phenotype 1
    g1 = haplotypes[[0, 1, 1, 2]]  # phenotype 1
    g2 = haplotypes[[0, 1, 2, 2]]  # phenotype 1
    g3 = haplotypes[[0, 0, 2, 2]]  # phenotype 2
    genotypes = np.array([g0, g1, g2, g3])

    t0 = genotypes[[0, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 10:0
    t1 = genotypes[[3, 2, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t2 = genotypes[[0, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 10:0
    t3 = genotypes[[3, 3, 3, 3, 3, 3, 3, 2, 1, 2]]  # 3:7
    trace = GenotypeMultiTrace(
        genotypes=np.array([t0, t1, t2, t3]), llks=np.ones((4, 10))
    )
    chain_modes = [dist.mode_phenotype() for dist in trace.chain_posteriors()]
    actual = str(obj(chain_modes))
    assert actual == expect


@pytest.mark.parametrize(
    "obj,expect",
    [
        pytest.param(
            filters.SampleChainPhenotypeCNVFilter(threshold=0.99), ".", id="NA"
        ),
        pytest.param(
            filters.SampleChainPhenotypeCNVFilter(threshold=0.8), "PASS", id="pass"
        ),
        pytest.param(
            filters.SampleChainPhenotypeCNVFilter(threshold=0.6), "cnv60", id="fail"
        ),
    ],
)
def test_SampleChainPhenotypeCNVFilter(obj, expect):
    haplotypes = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [1, 2],
        ]
    )

    g0 = haplotypes[[0, 0, 1, 2]]  # phenotype 1
    g1 = haplotypes[[0, 1, 1, 2]]  # phenotype 1
    g2 = haplotypes[[0, 1, 2, 2]]  # phenotype 1
    g3 = haplotypes[[0, 0, 2, 3]]  # phenotype 2
    g4 = haplotypes[[0, 2, 3, 4]]  # phenotype 3
    genotypes = np.array([g0, g1, g2, g3, g4])

    t0 = genotypes[[3, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t1 = genotypes[[3, 2, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t2 = genotypes[[0, 3, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t3 = genotypes[[3, 3, 4, 4, 4, 3, 4, 4, 4, 4]]  # 3:7
    trace = GenotypeMultiTrace(
        genotypes=np.array([t0, t1, t2, t3]), llks=np.ones((4, 10))
    )
    chain_modes = [dist.mode_phenotype() for dist in trace.chain_posteriors()]
    actual = str(obj(chain_modes))
    assert actual == expect
