import numpy as np
import pytest

from mchap.testing import simulate_reads
from mchap.calling.mcmc import gibbs_options, mh_options
from mchap.calling.classes import CallingMCMC
from mchap.calling.exact import genotype_likelihoods, genotype_posteriors


@pytest.mark.parametrize(
    "prior",
    [None, "flat_freqs", "rand_freqs"],
)
@pytest.mark.parametrize(
    "seed",
    [11, 42, 13, 0, 12234, 213, 45436, 1312, 374645],
)
def test_gibbs_mh_transition_equivalence(seed, prior):
    """Tests that transition matrices of gibbs and
    metropolis-hastings methods are equivalent.
    """
    # simulate haplotypes and reads
    np.random.seed(seed)
    inbreeding = np.random.rand()
    n_pos = np.random.randint(3, 13)
    n_haps = np.random.randint(2, 20)
    n_reads = np.random.randint(2, 15)
    ploidy = np.random.randint(2, 9)
    haplotypes = np.random.randint(0, 2, size=(n_haps, n_pos))
    # de-duplicate haplotypes
    haplotypes = np.unique(haplotypes, axis=0)
    n_haps = len(haplotypes)
    if prior == "rand_freqs":
        prior_frequencies = np.random.rand(n_haps)
        prior_frequencies /= prior_frequencies.sum()
        prior = (inbreeding, prior_frequencies)
    elif prior == "flat_freqs":
        prior_frequencies = None
        prior = (inbreeding, prior_frequencies)
    elif prior is None:
        prior = None
    genotype_alleles = np.random.randint(0, n_haps, size=ploidy)
    genotype_alleles.sort()
    genotype_haplotypes = haplotypes[genotype_alleles]
    reads = simulate_reads(
        genotype_haplotypes,
        n_alleles=np.full(n_pos, 2, int),
        n_reads=n_reads,
    )
    read_counts = np.random.randint(1, 10, size=n_reads)

    # single allele to vary
    variable_allele = np.random.randint(0, ploidy)

    # gibbs transition probabilities for the variable allele
    # these should be equivalent to the long run behavior of
    # the MH algorithm
    gibbs_llks_array = np.zeros(n_haps)
    gibbs_lpriors_array = np.zeros(n_haps)
    gibbs_probabilities_array = np.zeros(n_haps)
    gibbs_options(
        genotype_alleles=genotype_alleles,
        variable_allele=variable_allele,
        haplotypes=haplotypes,
        reads=reads,
        read_counts=read_counts,
        prior=prior,
        llks_array=gibbs_llks_array,
        lpriors_array=gibbs_lpriors_array,
        probabilities_array=gibbs_probabilities_array,
    )
    gibbs_long_run = gibbs_probabilities_array.copy()

    # calculate the long run behavior of the MH approach
    # by evaluating the transition probabilities for each
    # possible initial allele
    mh_llks_array = np.zeros(n_haps)
    mh_lpriors_array = np.zeros(n_haps)
    mh_probabilities_array = np.zeros(n_haps)
    mh_matrix = np.zeros((n_haps, n_haps))
    for a in range(n_haps):
        genotype_alleles[variable_allele] = a
        mh_options(
            genotype_alleles=genotype_alleles,
            variable_allele=variable_allele,
            haplotypes=haplotypes,
            reads=reads,
            read_counts=read_counts,
            prior=prior,
            llks_array=mh_llks_array,
            lpriors_array=mh_lpriors_array,
            probabilities_array=mh_probabilities_array,
        )
        mh_matrix[a, :] = mh_probabilities_array.copy()
        # the transition probabilities given a single allele
        # should not be the same as the gibbs approach
        assert any(gibbs_probabilities_array != mh_probabilities_array)

    # get long run behavior of MH approach
    mh_matrix = np.linalg.matrix_power(mh_matrix, 1000)
    mh_long_run = mh_matrix[0].copy()

    # assert gibbs and MH longruns are equivalent
    np.testing.assert_array_almost_equal(gibbs_long_run, mh_long_run)


@pytest.mark.parametrize(
    "seed,inbred,prior",
    [
        (11, False, None),
        (42, False, None),
        (11, False, "flat_freqs"),
        (42, True, "flat_freqs"),
        (13, False, "rand_freqs"),
        (0, True, "rand_freqs"),
        (12234, False, "flat_freqs"),
        (213, True, "flat_freqs"),
        (4536, False, "rand_freqs"),
        (2345, True, "rand_freqs"),
    ],
)
def test_gibbs_mh_mcmc_equivalence(seed, inbred, prior):
    np.random.seed(seed)
    inbreeding = np.random.rand() * inbred
    n_pos = np.random.randint(3, 13)
    n_haps = np.random.randint(2, 20)
    n_reads = np.random.randint(2, 15)
    ploidy = np.random.randint(2, 6)
    haplotypes = np.random.randint(0, 2, size=(n_haps, n_pos))
    # de-duplicate haplotypes
    haplotypes = np.unique(haplotypes, axis=0)
    n_haps = len(haplotypes)
    if prior == "rand_freqs":
        prior_frequencies = np.random.rand(n_haps)
        prior_frequencies /= prior_frequencies.sum()
        prior = (inbreeding, prior_frequencies)
    elif prior == "flat_freqs":
        prior_frequencies = None
        prior = (inbreeding, prior_frequencies)
    elif prior is None:
        prior = None
    genotype_alleles = np.random.randint(0, n_haps, size=ploidy)
    genotype_alleles.sort()
    genotype_haplotypes = haplotypes[genotype_alleles]
    reads = simulate_reads(
        genotype_haplotypes,
        n_alleles=np.full(n_pos, 2, int),
        n_reads=n_reads,
    )
    read_counts = np.random.randint(1, 10, size=n_reads)

    steps = 5000
    burn = 1000

    # Gibbs step
    gibbs_posterior = (
        CallingMCMC(
            ploidy=ploidy,
            haplotypes=haplotypes,
            prior=prior,
            steps=steps,
            random_seed=seed,
            step_type="Gibbs",
        )
        .fit(reads, read_counts)
        .burn(burn)
        .posterior()
        .as_array(n_haps)
    )

    # Metropolis-Hastings step
    mh_posterior = (
        CallingMCMC(
            ploidy=ploidy,
            haplotypes=haplotypes,
            prior=prior,
            steps=steps,
            random_seed=seed,
            step_type="Metropolis-Hastings",
        )
        .fit(reads, read_counts)
        .burn(burn)
        .posterior()
        .as_array(n_haps)
    )

    # Exact method
    llks = genotype_likelihoods(
        reads=reads,
        read_counts=read_counts,
        ploidy=ploidy,
        haplotypes=haplotypes,
    )
    exact_posterior = genotype_posteriors(
        llks,
        ploidy=ploidy,
        n_alleles=len(haplotypes),
        prior=prior,
    )

    np.testing.assert_array_almost_equal(gibbs_posterior, mh_posterior, decimal=2)
    np.testing.assert_array_almost_equal(gibbs_posterior, exact_posterior, decimal=2)
    np.testing.assert_array_almost_equal(mh_posterior, exact_posterior, decimal=2)
