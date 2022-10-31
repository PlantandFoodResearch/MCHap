import numpy as np
import pytest

from mchap.jitutils import increment_genotype, comb_with_replacement
from mchap.pedigree.prior import trio_log_pmf


@pytest.mark.parametrize(
    "seed",
    np.arange(50),
)
def test_trio_log_pmf__sum_to_one(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    ploidy_p = np.random.randint(2, 7)
    ploidy_q = np.random.randint(2, 7)
    parent_p = np.random.randint(n_alleles, size=ploidy_p)
    parent_q = np.random.randint(n_alleles, size=ploidy_q)
    tau_p = np.random.randint(1, ploidy_p)
    tau_q = np.random.randint(1, ploidy_q)
    error_p = np.random.rand()
    error_q = np.random.rand()
    inbreeding = np.random.rand()

    ploidy = tau_p + tau_q
    n_genotypes = comb_with_replacement(n_alleles, ploidy)

    total_prob = 0.0
    genotype = np.zeros(ploidy, int)
    for _ in range(n_genotypes):
        prob = np.exp(
            trio_log_pmf(
                progeny=genotype,
                parent_p=parent_p,
                parent_q=parent_q,
                tau_p=tau_p,
                tau_q=tau_q,
                lambda_p=0.0,
                lambda_q=0.0,
                error_p=error_p,
                error_q=error_q,
                inbreeding=inbreeding,
                n_alleles=n_alleles,
            )
        )
        total_prob += prob
        increment_genotype(genotype)
    np.testing.assert_almost_equal(total_prob, 1.0)


@pytest.mark.parametrize(
    "seed",
    np.arange(50),
)
@pytest.mark.parametrize(
    "use_lambda_p,use_lambda_q",
    [[True, False], [False, True], [True, True]],
)
def test_trio_log_pmf__sum_to_one_lambda(seed, use_lambda_p, use_lambda_q):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    ploidy_p = np.random.randint(2, 4)
    ploidy_q = np.random.randint(2, 4)
    parent_p = np.random.randint(n_alleles, size=ploidy_p)
    parent_q = np.random.randint(n_alleles, size=ploidy_q)
    tau_p, tau_q = 2, 2
    lambda_p = np.random.rand() if use_lambda_p else 0.0
    lambda_q = np.random.rand() if use_lambda_q else 0.0
    error_p = np.random.rand()
    error_q = np.random.rand()
    inbreeding = np.random.rand()

    ploidy = tau_p + tau_q
    n_genotypes = comb_with_replacement(n_alleles, ploidy)

    total_prob = 0.0
    genotype = np.zeros(ploidy, int)
    for _ in range(n_genotypes):
        prob = np.exp(
            trio_log_pmf(
                progeny=genotype,
                parent_p=parent_p,
                parent_q=parent_q,
                tau_p=tau_p,
                tau_q=tau_q,
                lambda_p=lambda_p,
                lambda_q=lambda_q,
                error_p=error_p,
                error_q=error_q,
                inbreeding=inbreeding,
                n_alleles=n_alleles,
            )
        )
        total_prob += prob
        increment_genotype(genotype)
    np.testing.assert_almost_equal(total_prob, 1.0)
