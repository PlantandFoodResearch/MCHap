import numpy as np
import pytest

from mchap.assemble.likelihood import log_likelihood
from mchap.assemble import mcmc
from mchap.jitutils import seed_numba
from mchap.testing import simulate_reads
from mchap.encoding import integer
from mchap import mset


def test_point_beta_probabilities():

    probs = mcmc._point_beta_probabilities(6, 1, 1)
    assert np.allclose(probs.sum(), 1.0)
    assert np.allclose(probs[0], probs)

    probs1 = mcmc._point_beta_probabilities(6, 1, 3)
    probs2 = mcmc._point_beta_probabilities(6, 3, 1)
    assert np.allclose(probs1.sum(), 1.0)
    assert np.allclose(probs2.sum(), 1.0)
    assert np.allclose(probs1, probs2[::-1])


def test_read_mean_dist():
    reads = np.array(
        [
            [[0.9, 0.1], [0.8, 0.2], [0.8, 0.2]],
            [[0.9, 0.1], [0.8, 0.2], [0.8, 0.2]],
            [[0.9, 0.1], [0.2, 0.8], [np.nan, np.nan]],
            [[0.9, 0.1], [0.2, 0.8], [np.nan, np.nan]],
        ]
    )
    expect = np.array(
        [
            [0.9, 0.1],
            [0.5, 0.5],
            [0.8, 0.2],
        ]
    )
    actual = mcmc._read_mean_dist(reads)
    np.testing.assert_array_equal(actual, expect)


def test_homozygosity_probabilities():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 1, 1, 1, 1],
        ]
    )
    ploidy = len(haplotypes)
    n_alleles = np.array([2] * 6)

    # 16 reads
    reads = simulate_reads(
        haplotypes,
        n_reads=16,
        uniform_sample=True,
        errors=False,
    )
    actual = mcmc._homozygosity_probabilities(reads, n_alleles, ploidy) > 0.999
    expect = np.zeros((6, 2), dtype=bool)
    np.testing.assert_array_equal(actual, expect)

    # 32 reads
    reads = simulate_reads(
        haplotypes,
        n_reads=32,
        uniform_sample=True,
        errors=False,
    )
    actual = mcmc._homozygosity_probabilities(reads, n_alleles, ploidy) > 0.999
    expect = np.array(
        [
            [True, False],
            [False, False],
            [False, False],
            [False, False],
            [False, True],
            [False, True],
        ],
        dtype=bool,
    )
    np.testing.assert_array_equal(actual, expect)


def test_DenovoMCMC__zero_reads():
    n_chains = 2
    ploidy, n_base = 4, 6
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([2] * n_base)
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

    # zero reads
    reads = np.empty((0, n_base, 2), dtype=float)

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()
    assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
    assert posterior.probabilities[0] < 0.05


def test_DenovoMCMC__zero_snps():
    n_chains = 2
    ploidy, n_base = 4, 0  # no snps in locus
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([], int)  # no snps in locus
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

    reads = np.empty((10, n_base, 0), dtype=float)

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()
    assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, 0)
    assert np.all(np.isnan(trace.llks))
    assert posterior.probabilities[0] == 1  # no variability


def test_DenovoMCMC__zero_reads_or_snps():
    n_chains = 2
    ploidy, n_base = 4, 0  # no snps in locus
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([], int)  # no snps in locus
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

    reads = np.empty((0, n_base, 0), dtype=float)

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()
    assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, 0)
    assert np.all(np.isnan(trace.llks))
    assert posterior.probabilities[0] == 1  # no variability


def test_DenovoMCMC__all_nans():
    n_chains = 2
    ploidy, n_base = 4, 6
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([2] * n_base)
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

    # high read depth
    reads = np.empty((10, n_base, 2), dtype=float)
    reads[:] = np.nan

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()
    assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
    assert posterior.probabilities[0] < 0.05


def test_DenovoMCMC__nans_ignored():
    # nan-reads should have no affect on mcmc
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    n_chains = 2
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([2] * n_base)
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

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
    haplotypes = np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    n_chains = 2
    ploidy, n_base = haplotypes.shape
    n_steps = 100
    n_burn = 50
    n_alleles = np.array([2] * n_base)
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

    # high read depth
    reads = simulate_reads(
        haplotypes,
        n_reads=40,
        errors=False,
        qual=(60, 60),
    )

    trace = model.fit(reads).burn(n_burn)
    posterior = trace.posterior()

    assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
    # nan likelihoods for non-variable chain
    assert np.isnan(trace.llks).all()
    assert len(posterior.genotypes) == 1
    np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])
    assert posterior.probabilities[0] == 1.0


def test_DenovoMCMC__diploid():

    np.random.seed(42)

    haplotypes = np.array(
        [
            [0, 0, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 0],
        ]
    )
    n_chains = 2
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([2] * n_base)
    model = mcmc.DenovoMCMC(
        ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
    )

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
        assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
        assert posterior.probabilities[0] > 0.90
        np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])


def test_DenovoMCMC__tetraploid():

    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    n_chains = 2
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_burn = 500
    n_alleles = np.array([2] * n_base)
    model = mcmc.DenovoMCMC(
        ploidy=ploidy,
        n_alleles=n_alleles,
        steps=n_steps,
        chains=n_chains,
        random_seed=42,
    )

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
        assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
        assert posterior.probabilities[0] > 0.90
        np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])
        assert np.any((trace.genotypes[0] != trace.genotypes[1]))

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
        assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
        assert 0.30 < posterior.probabilities[0] < 0.90
        np.testing.assert_array_equal(haplotypes, posterior.genotypes[0])
        assert np.any((trace.genotypes[0] != trace.genotypes[1]))

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
        assert trace.genotypes.shape == (n_chains, n_steps - n_burn, ploidy, n_base)
        assert posterior.probabilities[0] < 0.30
        assert np.any((trace.genotypes[0] != trace.genotypes[1]))


def test_DenovoMCMC__seed():

    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 1, 1, 1],
            [0, 1, 0, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    n_chains = 2
    ploidy, n_base = haplotypes.shape
    n_steps = 1000
    n_alleles = np.array([2] * n_base)

    # medium read depth
    reads = simulate_reads(
        haplotypes,
        n_reads=16,
        uniform_sample=True,
        errors=False,
        qual=(60, 60),
    )

    model_1 = mcmc.DenovoMCMC(
        ploidy, n_alleles, steps=n_steps, chains=n_chains, random_seed=42
    )
    model_2 = mcmc.DenovoMCMC(
        ploidy, n_alleles, steps=n_steps, chains=n_chains, random_seed=33
    )
    model_3 = mcmc.DenovoMCMC(
        ploidy, n_alleles, steps=n_steps, chains=n_chains, random_seed=42
    )

    trace_1 = model_1.fit(reads)
    trace_2 = model_2.fit(reads)
    trace_3 = model_3.fit(reads)

    # compair among runs
    assert np.any((trace_1.genotypes != trace_2.genotypes))
    np.testing.assert_array_equal(trace_1.genotypes, trace_3.genotypes)

    # compair within runs (among chains)
    assert np.any((trace_1.genotypes[0] != trace_1.genotypes[1]))
    assert np.any((trace_2.genotypes[0] != trace_2.genotypes[1]))


def test_DenovoMCMC__fuzz():
    for _ in range(10):
        n_chains = np.random.randint(1, 4)
        ploidy = np.random.randint(2, 5)
        n_base = np.random.randint(4, 8)
        n_steps = np.random.randint(100, 400)
        n_reads = np.random.randint(1, 50)

        haplotypes = np.random.choice(
            [0, 1, 2],  # small chance of triallelic base
            p=[0.45, 0.45, 0.1],
            size=(ploidy, n_base),
        )
        n_alleles = np.array([np.max(haplotypes) + 1] * n_base)
        model = mcmc.DenovoMCMC(
            ploidy=ploidy, n_alleles=n_alleles, steps=n_steps, chains=n_chains
        )
        reads = simulate_reads(haplotypes, n_reads=n_reads)
        trace = model.fit(reads)
        assert trace.genotypes.shape == (n_chains, n_steps, ploidy, n_base)
        assert trace.llks.shape == (n_chains, n_steps)


@pytest.mark.parametrize(
    "temperatures",
    [
        [1.0],
        [0.1, 1.0],
        [0.01, 0.1, 1.0],
        [0.001, 0.01, 0.1, 1.0],
    ],
)
def test_DenovoMCMC__temperatures_bias(temperatures):
    """Test to ensure the implementation of parallel-tempering does
    not bias the MCMC results when tempering is not needed.
    """
    reads = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.9, 0.1], [0.1, 0.9]],
            [[0.9, 0.1], [0.9, 0.1]],
            [[0.9, 0.1], [0.9, 0.1]],
        ]
    )
    genotypes = np.array(
        [
            [[0, 0], [0, 0]],  # 2
            [[0, 0], [0, 1]],  # 1:1
            [[0, 0], [1, 0]],  # 1:1
            [[0, 0], [1, 1]],  # 1:1
            [[0, 1], [0, 1]],  # 2
            [[0, 1], [1, 0]],  # 1:1
            [[0, 1], [1, 1]],  # 1:1
            [[1, 0], [1, 0]],  # 2
            [[1, 0], [1, 1]],  # 1:1
            [[1, 1], [1, 1]],  # 2
        ],
        dtype=np.int8,
    )

    # calculate exact posteriors
    llks = np.array([log_likelihood(reads, g) for g in genotypes])
    priors = np.array([1, 2, 2, 2, 1, 2, 2, 1, 2, 1])
    priors = priors / priors.sum()
    exact_posteriors = np.exp(llks + np.log(priors))
    exact_posteriors = exact_posteriors / exact_posteriors.sum()

    # simulate posteriors
    model = mcmc.DenovoMCMC(
        ploidy=2,
        n_alleles=[2, 2],
        steps=50000,
        chains=1,
        random_seed=11,
        temperatures=temperatures,
    )
    trace = model.fit(reads)
    posterior = trace.burn(100).posterior()
    simulation_posteriors = {g.tobytes(): 0 for g in genotypes}
    simulation_posteriors.update(
        {g.tobytes(): p for g, p in zip(posterior.genotypes, posterior.probabilities)}
    )
    simulation_posteriors = [simulation_posteriors[g.tobytes()] for g in genotypes]

    # check posteriors are similar
    np.testing.assert_array_almost_equal(
        exact_posteriors,
        simulation_posteriors,
        decimal=2,
    )


def test_DenovoMCMC__temperatures_submode():
    """This test is taken from real world example in which the mcmc
    can become stuck in a sub-optimal alternative mode when parallel
    tempering is not used. Note that the optimal mode is not necessarily
    the true genotype and the data there may be additional haplotypes due
    to duplication.
    """
    read_counts = [
        ("000000", 22),
        ("-00000", 3),
        ("0000-0", 2),
        ("00000-", 1),
        ("000-00", 1),
        ("001000", 16),
        ("-01000", 2),
        ("001---", 2),
        ("--1000", 3),
        ("000110", 39),
        ("-00110", 1),
        ("----10", 1),
        ("100101", 3),
        ("-00101", 3),
        ("----01", 1),
        ("110100", 1),
        ("110---", 2),
        ("1101--", 1),
        ("11----", 1),
        ("0--000", 1),
        ("---000", 3),
        ("----00", 4),
        ("0-----", 2),
        ("1-----", 2),
        ("001110", 1),
    ]
    strings, counts = zip(*read_counts)
    calls = integer.from_strings(strings)
    counts = np.array(list(counts))
    dists = integer.as_probabilistic(
        calls,
        n_alleles=2,
        p=0.999,
    )
    reads = mset.repeat(dists, counts)
    n_alleles = np.array([2] * 6)

    alt_mode = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 1],
        ]
    )

    opt_mode = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 0],
        ]
    )

    # initialise all chains within sub-optimal mode
    initial = np.tile(alt_mode, (2, 1, 1))

    # without tempering
    model = mcmc.DenovoMCMC(
        ploidy=6, n_alleles=n_alleles, steps=1100, temperatures=[1.0], random_seed=1
    )
    posterior = model.fit(reads, initial=initial).burn(100).posterior()
    np.testing.assert_array_equal(posterior.genotypes[0], alt_mode)  # sub-optimal mode
    assert posterior.probabilities[0] > 0.95

    # with tempering
    model = mcmc.DenovoMCMC(
        ploidy=6,
        n_alleles=n_alleles,
        steps=1100,
        temperatures=[0.1, 1.0],
        random_seed=1,
    )
    posterior = model.fit(reads, initial=initial).burn(100).posterior()
    np.testing.assert_array_equal(posterior.genotypes[0], opt_mode)  # optimal mode
    assert posterior.probabilities[0] > 0.95
