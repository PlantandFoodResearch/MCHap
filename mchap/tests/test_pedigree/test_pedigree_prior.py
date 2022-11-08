import numpy as np
import pytest

from mchap.jitutils import increment_genotype, comb_with_replacement
from mchap.calling.utils import allelic_dosage
from mchap.pedigree.prior import (
    parental_copies,
    dosage_permutations,
    initial_dosage,
    increment_dosage,
    duplicate_permutations,
    gamete_log_pmf,
    second_gamete_log_pmf,
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
    np.arange(10),
)
def test_log_gamete_pmf__sum_to_one(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    gamete_ploidy = np.random.randint(1, 3)
    parent_ploidy = np.random.randint(2, 4)
    parent_genotype = np.random.randint(n_alleles, size=parent_ploidy)
    n_gametes = comb_with_replacement(n_alleles, gamete_ploidy)
    total_prob = 0.0
    gamete_genotype = np.zeros(gamete_ploidy, int)
    for _ in range(n_gametes):
        gamete_dosage = allelic_dosage(gamete_genotype)
        parent_dosage = parental_copies(parent_genotype, gamete_genotype)
        prob = np.exp(
            gamete_log_pmf(
                gamete_dose=gamete_dosage,
                gamete_ploidy=gamete_ploidy,
                parent_dose=parent_dosage,
                parent_ploidy=parent_ploidy,
                gamete_lambda=0.0,
            )
        )
        total_prob += prob
        increment_genotype(gamete_genotype)
    np.testing.assert_almost_equal(total_prob, 1.0)


@pytest.mark.parametrize(
    "seed",
    np.arange(10),
)
def test_log_gamete_pmf__sum_to_one_lambda(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    gamete_ploidy = 2
    parent_ploidy = np.random.randint(2, 4)
    parent_genotype = np.random.randint(n_alleles, size=parent_ploidy)
    n_gametes = comb_with_replacement(n_alleles, gamete_ploidy)
    total_prob = 0.0
    lambda_ = np.random.rand()
    gamete_genotype = np.zeros(gamete_ploidy, int)
    for _ in range(n_gametes):
        gamete_dosage = allelic_dosage(gamete_genotype)
        parent_dosage = parental_copies(parent_genotype, gamete_genotype)
        prob = np.exp(
            gamete_log_pmf(
                gamete_dose=gamete_dosage,
                gamete_ploidy=gamete_ploidy,
                parent_dose=parent_dosage,
                parent_ploidy=parent_ploidy,
                gamete_lambda=lambda_,
            )
        )
        total_prob += prob
        increment_genotype(gamete_genotype)
    np.testing.assert_almost_equal(total_prob, 1.0)


def test_log_gamete_pmf__raise_on_non_diploid_lambda():
    with pytest.raises(
        ValueError, match="Lambda parameter is only supported for diploid gametes"
    ):
        gamete_log_pmf(
            gamete_dose=np.array([2, 1, 0]),
            gamete_ploidy=3,
            parent_dose=np.array([2, 2, 2]),
            parent_ploidy=6,
            gamete_lambda=0.01,
        )


@pytest.mark.parametrize(
    "seed",
    np.arange(10),
)
def test_second_gamete_log_pmf__sum_to_one(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    inbreeding = np.random.rand()
    constant_dose = np.zeros(n_alleles, int)
    const_ploidy = np.random.randint(1, 4)
    gamete_ploidy = np.random.randint(1, 4)
    for _ in range(const_ploidy):
        a = np.random.randint(n_alleles)
        constant_dose[a] += 1
    constraint = np.full(n_alleles, gamete_ploidy, int)
    n_gametes = comb_with_replacement(n_alleles, gamete_ploidy)

    total_prob = 0.0
    gamete_dose = initial_dosage(gamete_ploidy, constraint)
    for i in range(0, n_gametes):
        if i:
            increment_dosage(gamete_dose, constraint)
        prob = np.exp(
            second_gamete_log_pmf(gamete_dose, constant_dose, n_alleles, inbreeding)
        )
        total_prob += prob
    np.testing.assert_almost_equal(total_prob, 1.0)


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
    "use_lambda_p,use_lambda_q",
    [[True, False], [False, True], [True, True]],
)
@pytest.mark.parametrize(
    "seed",
    np.arange(50),
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
