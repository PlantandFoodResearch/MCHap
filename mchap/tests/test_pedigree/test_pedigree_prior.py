import numpy as np
import pytest

from mchap.jitutils import increment_genotype, comb_with_replacement
from mchap.pedigree.prior import (
    parental_copies,
    dosage_permutations,
    initial_dosage,
    increment_dosage,
    duplicate_permutations,
    trio_log_pmf,
)


@pytest.mark.parametrize(
    "parent, progeny, expect",
    [
        ([0, 0, 0, 0], [0, 0], [4, 0]),
        ([0, 1, 1, 2], [0, 2], [1, 1]),
        ([0, 1, 2, 3, 4, 5], [6, 7, 8], [0, 0, 0]),
        ([0, 1], [1, 1], [1, 0]),
    ],
)
def test_parental_copies(parent, progeny, expect):
    progeny = np.array(progeny)
    parent = np.array(parent)
    expect = np.array(expect)
    observed = parental_copies(parent, progeny)
    np.testing.assert_array_equal(observed, expect)


@pytest.mark.parametrize(
    "gamete_dosage, parent_dosage, expect",
    [
        ([2, 0], [2, 0], 1),
        ([2, 0], [3, 0], 3),
        ([2, 0], [4, 0], 6),
        ([1, 1], [1, 0], 0),
        ([1, 1], [1, 1], 1),
        ([1, 1], [2, 1], 2),
        ([1, 1], [2, 2], 4),
        ([2, 1, 0], [2, 2, 0], 2),
        ([2, 1, 0], [3, 2, 0], 6),
    ],
)
def test_dosage_permutations(gamete_dosage, parent_dosage, expect):
    gamete_dosage = np.array(gamete_dosage)
    parent_dosage = np.array(parent_dosage)
    observed = dosage_permutations(gamete_dosage, parent_dosage)
    assert observed == expect


@pytest.mark.parametrize(
    "ploidy, constraint, expect",
    [
        (2, [2, 0, 2, 0], [2, 0, 0, 0]),
        (2, [1, 2, 1, 0], [1, 1, 0, 0]),
        (3, [1, 2, 1, 0], [1, 2, 0, 0]),
    ],
)
def test_initial_dosage(ploidy, constraint, expect):
    constraint = np.array(constraint)
    expect = np.array(expect)
    observed = initial_dosage(ploidy, constraint)
    np.testing.assert_array_equal(observed, expect)


def test_initial_dosage__raise_on_ploidy():
    ploidy = 2
    constraint = np.array([1, 0, 0, 0])
    with pytest.raises(ValueError, match="Ploidy does not fit within constraint"):
        initial_dosage(ploidy, constraint)


@pytest.mark.parametrize(
    "dosage, constraint, expect",
    [
        ([2, 0, 0, 0], [2, 0, 2, 0], [1, 0, 1, 0]),
        ([1, 0, 1, 0], [2, 0, 2, 0], [0, 0, 2, 0]),
        ([1, 1, 0, 0], [3, 1, 2, 0], [1, 0, 1, 0]),
        ([1, 0, 1, 0], [3, 1, 2, 0], [0, 1, 1, 0]),
    ],
)
def test_increment_dosage(dosage, constraint, expect):
    dosage = np.array(dosage)
    expect = np.array(expect)
    constraint = np.array(constraint)
    increment_dosage(dosage, constraint)
    np.testing.assert_array_equal(dosage, expect)


def test_increment_dosage__raise_on_final():
    dosage = np.array([0, 0, 2, 0])
    constraint = np.array([2, 0, 2, 0])
    with pytest.raises(ValueError, match="Final dosage"):
        increment_dosage(dosage, constraint)


@pytest.mark.parametrize(
    "gamete_dosage, parent_dosage, expect",
    [
        ([1, 1], [1, 1], 0),
        ([1, 1], [2, 0], 0),
        ([2, 0], [1, 0], 1),
        ([2, 0], [2, 0], 2),
    ],
)
def test_duplicate_permutations(gamete_dosage, parent_dosage, expect):
    gamete_dosage = np.array(gamete_dosage)
    parent_dosage = np.array(parent_dosage)
    observed = duplicate_permutations(gamete_dosage, parent_dosage)
    assert expect == observed


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
