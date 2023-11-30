import numpy as np
import pytest

from mchap.testing import simulate_reads
from mchap.jitutils import (
    increment_genotype,
    comb_with_replacement,
    sum_log_probs,
    genotype_alleles_as_index,
    add_log_prob,
)
from mchap.pedigree.prior import trio_log_pmf
from mchap.pedigree.likelihood import log_likelihood_alleles_cached
from mchap.pedigree.classes import PedigreeCallingMCMC, PedigreeAllelesMultiTrace


@pytest.mark.parametrize(
    "read_depth, step_type, seed, tolerance",
    [
        (0, "Gibbs", 0, 0.02),
        (0, "Metropolis-Hastings", 0, 0.03),
        (4, "Gibbs", 0, 0.02),
        (4, "Metropolis-Hastings", 0, 0.03),
        (10, "Gibbs", 0, 0.01),
        (10, "Metropolis-Hastings", 0, 0.015),
        (40, "Gibbs", 0, 0.01),
        (40, "Metropolis-Hastings", 0, 0.015),
        (100, "Gibbs", 0, 0.0015),
        (100, "Metropolis-Hastings", 0, 0.003),
    ],
)
def test_PedigreeCallingMCMC__exact(read_depth, step_type, seed, tolerance):
    """Test that MCMC approximates the true posterior as calculated for a tractable example.

    Note
    ----
    Lower read depth results in more genotypes to check and greater variability.
    Metropolis-Hastings is less statistically efficient resulting in greater variability for
    same number of steps, especially with sparse posteriors resulting from higher read depths.
    """
    # pedigree details
    sample_parent = np.array(
        [
            [-1, -1],
            [-1, -1],
            [0, 1],
            [-1, 2],  # unknown parent
        ]
    )
    gamete_tau = np.array(
        [
            [1, 1],
            [2, 2],
            [2, 2],
            [2, 2],
        ]
    )
    gamete_lambda = np.array([[0.0, 0.0], [0.01, 0.01], [0.345, 0.01], [0.01, 0.01]])
    gamete_error = np.array([[0.01, 0.01], [0.01, 0.01], [0.01, 0.01], [0.01, 0.01]])
    true_genotype = np.array(
        [
            [0, 1, -2, -2],
            [0, 0, 0, 2],
            [0, 1, 1, 2],
            [0, 0, 1, 2],
        ]
    )
    sample_ploidy = gamete_tau.sum(axis=-1)

    # haplotypes (few to limit complexity)
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 1, 1, 0, 0, 1, 1],
        ]
    )
    np.random.seed(seed)
    frequencies = np.random.rand(len(haplotypes))
    frequencies /= frequencies.sum()
    print("Frequencies:", frequencies)

    # simulate reads from genotypes
    n_samples = len(sample_parent)
    n_alleles, n_pos = haplotypes.shape
    sample_read_dists = np.empty((n_samples, read_depth, n_pos, 2))
    for i, g in enumerate(true_genotype):
        sample_read_dists[i] = simulate_reads(
            haplotypes[g],
            n_alleles=2,
            n_reads=read_depth,
            errors=False,
            uniform_sample=False,
        )
    sample_read_counts = np.ones((n_samples, read_depth), dtype=int)

    # arrays for storing exact posterior
    n_diploid_genotypes = comb_with_replacement(n_alleles, 2)
    n_tetraploid_genotypes = comb_with_replacement(n_alleles, 4)
    genotype_exact_posterior = np.full((4, n_tetraploid_genotypes), -np.inf)
    pedigree_exact_posterior = np.full(
        (
            n_diploid_genotypes,
            n_tetraploid_genotypes,
            n_tetraploid_genotypes,
            n_tetraploid_genotypes,
        ),
        -np.inf,
    )

    # scratch variables
    _, max_ploidy = true_genotype.shape
    dosage = np.zeros(max_ploidy, dtype=np.int64)
    dosage_p = np.zeros(max_ploidy, dtype=np.int64)
    dosage_q = np.zeros(max_ploidy, dtype=np.int64)
    gamete_p = np.zeros(max_ploidy, dtype=np.int64)
    gamete_q = np.zeros(max_ploidy, dtype=np.int64)
    constraint_p = np.zeros(max_ploidy, dtype=np.int64)
    constraint_q = np.zeros(max_ploidy, dtype=np.int64)

    # brute force exact posterior by iteration over all combinations of genotypes
    current_genotype = np.zeros((4, 4), int)
    current_genotype[0, 2:] = -2  # first sample is diploid
    for i_0 in range(n_diploid_genotypes):
        for i_1 in range(n_tetraploid_genotypes):
            for i_2 in range(n_tetraploid_genotypes):
                for i_3 in range(n_tetraploid_genotypes):
                    # calculate un-normalized posterior prob
                    log_prior = np.log(1.0)
                    log_like = np.log(1.0)
                    for j in range(n_samples):
                        # prior
                        p, q = sample_parent[j]
                        if p >= 0:
                            error_p = gamete_error[j, 0]
                            ploidy_p = sample_ploidy[p]
                        else:
                            error_p = 1.0
                            ploidy_p = 0
                        if q >= 0:
                            error_q = gamete_error[j, 1]
                            ploidy_q = sample_ploidy[q]
                        else:
                            error_q = 1.0
                            ploidy_q = 0
                        log_prior += trio_log_pmf(
                            current_genotype[j],
                            current_genotype[p],
                            current_genotype[q],
                            ploidy_p=ploidy_p,
                            ploidy_q=ploidy_q,
                            tau_p=gamete_tau[j, 0],
                            tau_q=gamete_tau[j, 1],
                            lambda_p=gamete_lambda[j, 0],
                            lambda_q=gamete_lambda[j, 1],
                            error_p=error_p,
                            error_q=error_q,
                            log_frequencies=np.log(frequencies),
                            dosage=dosage,
                            dosage_p=dosage_p,
                            dosage_q=dosage_q,
                            gamete_p=gamete_p,
                            gamete_q=gamete_q,
                            constraint_p=constraint_p,
                            constraint_q=constraint_q,
                        )
                        # likelihood
                        log_like += log_likelihood_alleles_cached(
                            reads=sample_read_dists[j],
                            read_counts=sample_read_counts[j],
                            haplotypes=haplotypes,
                            sample=j,
                            genotype_alleles=current_genotype[j, 0 : sample_ploidy[j]],
                            cache=None,
                        )
                    log_post = log_like + log_prior
                    # store un-normalized posterior per sample and total
                    genotype_exact_posterior[0, i_0] = add_log_prob(
                        genotype_exact_posterior[0, i_0], log_post
                    )
                    genotype_exact_posterior[1, i_1] = add_log_prob(
                        genotype_exact_posterior[1, i_1], log_post
                    )
                    genotype_exact_posterior[2, i_2] = add_log_prob(
                        genotype_exact_posterior[2, i_2], log_post
                    )
                    genotype_exact_posterior[3, i_3] = add_log_prob(
                        genotype_exact_posterior[3, i_3], log_post
                    )
                    pedigree_exact_posterior[i_0, i_1, i_2, i_3] = log_post
                    # increment/reset genotypes
                    increment_genotype(current_genotype[3])
                current_genotype[3] = 0
                increment_genotype(current_genotype[2])
            current_genotype[2] = 0
            increment_genotype(current_genotype[1])
        current_genotype[1] = 0
        increment_genotype(current_genotype[0, 0:2])

    # normalize posteriors
    denominator = sum_log_probs(pedigree_exact_posterior.ravel())
    genotype_exact_posterior = np.exp(genotype_exact_posterior - denominator)
    pedigree_exact_posterior = np.exp(pedigree_exact_posterior - denominator)

    # approximate with MCMC model
    model = PedigreeCallingMCMC(
        sample_ploidy=sample_ploidy,
        sample_inbreeding=np.zeros(n_samples),
        sample_parents=sample_parent,
        gamete_tau=gamete_tau,
        gamete_lambda=gamete_lambda,
        gamete_error=gamete_error,
        haplotypes=haplotypes,
        frequencies=frequencies,
        steps=3000,
        annealing=1000,
        chains=2,
        random_seed=seed,
        step_type=step_type,
    )
    trace = model.fit(sample_read_dists, sample_read_counts).burn(1000)

    # compare posteriors for individual genotypes
    for i in range(n_samples):
        post = trace.individual(i).posterior()
        for g, actual in zip(post.genotypes, post.probabilities):
            j = genotype_alleles_as_index(g)
            expect = genotype_exact_posterior[i, j]
            print(
                "Sample:",
                i,
                g,
                "tolerance:",
                tolerance,
                "expected:",
                expect.round(5),
                "actual:",
                actual.round(5),
            )
            assert np.allclose(expect, actual, atol=tolerance)


def test_PedigreeAllelesMultiTrace_burn():
    trace_0 = [
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
    ]
    trace_1 = [
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
    ]
    trace = PedigreeAllelesMultiTrace(np.array([trace_0, trace_1]))
    actual = trace.burn(2).genotypes
    expect = [trace_0[2:], trace_1[2:]]
    np.testing.assert_array_equal(expect, actual)


def test_PedigreeAllelesMultiTrace_individual():
    trace_0 = [
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
    ]
    trace_1 = [
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
    ]
    trace = PedigreeAllelesMultiTrace(np.array([trace_0, trace_1]))
    actual = trace.individual(2).genotypes
    expect = [np.array(trace_0)[:, 2], np.array(trace_1)[:, 2]]
    np.testing.assert_array_equal(expect, actual)


def test_PedigreeAllelesMultiTrace_incongruence():
    sample_ploidy = np.array([4, 4, 4])
    sample_parents = np.array(
        [
            [-1, -1],
            [0, -1],
            [0, 1],
        ]
    )
    gamete_tau = np.full((3, 2), 2, int)
    gamete_lambda = np.zeros((3, 2), float)
    trace_0 = [
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
    ]
    trace_1 = [
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 2]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
        [[0, 0, 0, 1], [0, 0, 0, 2], [0, 0, 1, 3]],
    ]
    trace = PedigreeAllelesMultiTrace(np.array([trace_0, trace_1]))
    expect = [0.0, 0.0, 3 / 10]
    actual = trace.incongruence(
        sample_ploidy, sample_parents, gamete_tau, gamete_lambda
    )
    np.testing.assert_array_equal(expect, actual)
