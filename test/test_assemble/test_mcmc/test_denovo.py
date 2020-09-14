import numpy as np
import numba
import pytest

from haplokit.assemble.mcmc import denovo
from haplokit.assemble.util import seed_numba
from haplokit.testing import simulate_reads


def test_point_beta_probabilities():

    probs = denovo._point_beta_probabilities(6, 1, 1)
    assert np.allclose(probs.sum(), 1.0)
    assert np.allclose(probs[0], probs)

    probs1 = denovo._point_beta_probabilities(6, 1, 3)
    probs2 = denovo._point_beta_probabilities(6, 3, 1)
    assert np.allclose(probs1.sum(), 1.0)
    assert np.allclose(probs2.sum(), 1.0)
    assert np.allclose(probs1, probs2[::-1])


def test_read_mean_dist():
    reads = np.array([
        [[0.9, 0.1],
        [0.8, 0.2],
        [0.8, 0.2]],
        [[0.9, 0.1],
        [0.8, 0.2],
        [0.8, 0.2]],
        [[0.9, 0.1],
        [0.2, 0.8],
        [np.nan, np.nan]],
        [[0.9, 0.1],
        [0.2, 0.8],
        [np.nan, np.nan]]
    ])
    expect = np.array([
        [0.9, 0.1],
        [0.5, 0.5],
        [0.8, 0.2],
    ])
    actual = denovo._read_mean_dist(reads)
    np.testing.assert_array_equal(actual, expect)


def test_homozygosity_probabilities():
    haplotypes = np.array([
        [0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1],
    ])
    ploidy = len(haplotypes)

    # 16 reads
    reads = simulate_reads(
        haplotypes, 
        n_reads=16, 
        uniform_sample=True,
        errors=False, 
    )
    actual = denovo._homozygosity_probabilities(reads, ploidy) > 0.999
    expect = np.zeros((6, 2), dtype=np.bool)
    np.testing.assert_array_equal(actual, expect)

    # 32 reads
    reads = simulate_reads(
        haplotypes, 
        n_reads=32, 
        uniform_sample=True,
        errors=False, 
    )
    actual = denovo._homozygosity_probabilities(reads, ploidy) > 0.999
    expect = np.array([
        [True, False],
        [False, False],
        [False, False],
        [False, False],
        [False, True],
        [False, True],
    ], dtype=np.bool)
    np.testing.assert_array_equal(actual, expect)


def test_DenovoMCMC__all_nans():
    ploidy, n_base = 4, 6
    n_steps = 1000
    n_burn = 500
    model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)

    # high read depth
    reads = np.empty((10, n_base, 2), dtype=np.float)
    reads[:] = np.nan

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()
    assert trace.genotypes.shape == (n_steps - n_burn, ploidy, n_base)
    assert posterior.probabilities[0] < 0.05



def test_DenovoMCMC__nans_ignored():
    # nan-reads should have no affect on mcmc
    haplotypes = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)

    reads1 = simulate_reads(
        haplotypes, 
        n_reads=16, 
        uniform_sample=True,
        errors=False, 
        qual=(60, 60),
    )

    nan_reads = np.empty((4, n_base, 2), dtype=reads1.dtype)
    nan_reads[:] = np.nan
    reads2 = np.concatenate([reads1, nan_reads])

    np.random.seed(42)
    seed_numba(42)
    trace1 = model.fit(reads1).burn(n_burn)

    np.random.seed(42)
    seed_numba(42)
    trace2 = model.fit(reads2).burn(n_burn)

    np.testing.assert_array_equal(trace1.genotypes, trace2.genotypes)


def test_DenovoMCMC__non_variable():
    haplotypes = np.array([
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1, 1],
    ])
    ploidy, n_base = haplotypes.shape
    n_steps = 100
    n_burn = 50
    model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)

    # high read depth
    reads = simulate_reads(
        haplotypes, 
        n_reads=40, 
        errors=False, 
        qual=(60, 60),
    )

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()

    assert trace.genotypes.shape == (n_steps - n_burn, ploidy, n_base)
    # nan likelihoods for non-variable chain
    assert np.isnan(trace.llks).all()
    assert len(posterior.genotypes) == 1
    np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])
    assert posterior.probabilities[0] == 1.0


def test_DenovoMCMC__diploid():

    np.random.seed(42)

    haplotypes = np.array([
        [0, 0, 0, 1, 1, 1],
        [1, 0, 0, 0, 0, 0],
    ])
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)

    reads = simulate_reads(
        haplotypes, 
        n_reads=2, 
        uniform_sample=True,
        errors=False, 
        qual=(60, 60),
    )

    for i in range(10):
        trace = model.fit(reads).burn(n_burn)
        posterior = trace.posterior()
        assert trace.genotypes.shape == (n_steps - n_burn, ploidy, n_base)
        assert posterior.probabilities[0] > 0.90
        np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])

def test_DenovoMCMC__tetraploid():

    np.random.seed(42)

    haplotypes = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)

    # high read depth
    reads = simulate_reads(
        haplotypes, 
        n_reads=40, 
        uniform_sample=True,
        errors=False, 
        qual=(60, 60),
    )
    for _ in range(10):
        trace = model.fit(reads).burn(n_burn)
        posterior = trace.posterior()
        assert trace.genotypes.shape == (n_steps - n_burn, ploidy, n_base)
        assert posterior.probabilities[0] > 0.90
        np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])

    # medium read depth
    reads = simulate_reads(
        haplotypes, 
        n_reads=16, 
        uniform_sample=True,
        errors=False, 
        qual=(60, 60),
    )
    for _ in range(10):
        trace = model.fit(reads).burn(n_burn)
        posterior = trace.posterior()
        assert trace.genotypes.shape == (n_steps - n_burn, ploidy, n_base)
        assert 0.30 < posterior.probabilities[0] < 0.90
        np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])

    # low read depth
    reads = simulate_reads(
        haplotypes, 
        n_reads=8, 
        uniform_sample=True,
        errors=False, 
        qual=(60, 60),
    )
    for _ in range(10):
        trace = model.fit(reads).burn(n_burn)
        posterior = trace.posterior()
        assert trace.genotypes.shape == (n_steps - n_burn, ploidy, n_base)
        assert posterior.probabilities[0] < 0.30


def test_DenovoMCMC__seed():

    haplotypes = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 1, 1, 1],
        [0, 1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1, 1],
    ])
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)
    # medium read depth
    reads = simulate_reads(
        haplotypes, 
        n_reads=16, 
        uniform_sample=True,
        errors=False, 
        qual=(60, 60),
    )

    model_1 = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps, random_seed=42)
    model_2 = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps, random_seed=33)
    model_3 = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps, random_seed=42)

    trace_1 = model_1.fit(reads)
    trace_2 = model_2.fit(reads)
    trace_3 = model_3.fit(reads)

    assert np.any((trace_1.genotypes != trace_2.genotypes))
    np.testing.assert_array_equal(trace_1.genotypes, trace_3.genotypes)


def test_DenovoMCMC__fuzz():
    for _ in range(10):
        ploidy = np.random.randint(2, 5)
        n_base = np.random.randint(4, 8)
        n_steps = np.random.randint(100, 400)
        n_reads = np.random.randint(1, 50)

        haplotypes = np.random.choice(
            [0, 1, 2],  # small chance of triallelic base
            p=[0.45, 0.45, 0.1], 
            size=(ploidy, n_base),
        )
        model = denovo.DenovoMCMC(ploidy=ploidy, steps=n_steps)
        reads = simulate_reads(haplotypes, n_reads=n_reads)
        trace = model.fit(reads)
        assert trace.genotypes.shape == (n_steps, ploidy, n_base)

