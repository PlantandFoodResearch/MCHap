import numpy as np
import pytest

from mchap.jitutils import increment_genotype, comb_with_replacement
from mchap.calling.utils import count_allele
from mchap.pedigree.prior import (
    set_allelic_dosage,
    set_parental_copies,
    dosage_permutations,
    set_initial_dosage,
    set_complimentary_gamete,
    increment_dosage,
    double_reduction_permutations,
    gamete_log_pmf,
    gamete_allele_log_pmf,
    gamete_const_log_pmf,
    trio_log_pmf,
    log_unknown_dosage_prior,
    log_unknown_const_prior,
)


@pytest.mark.parametrize(
    "parent, progeny, expect",
    [
        ([0, 0, 0, 0], [0, 0, -2, -2], [4, 0, 0, 0]),
        ([0, 1, 1, 2], [0, 2, -2, -2], [1, 1, 0, 0]),
        ([0, 1, 2, 3, 4, 5], [6, 7, 8, -2, -2, -2], [0, 0, 0, 0, 0, 0]),
        ([0, 1], [1, 1], [1, 0]),
    ],
)
def test_set_parental_copies(parent, progeny, expect):
    progeny = np.array(progeny)
    parent = np.array(parent)
    expect = np.array(expect)
    observed = np.zeros_like(progeny)
    set_parental_copies(parent, progeny, observed)
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
def test_set_initial_dosage(ploidy, constraint, expect):
    constraint = np.array(constraint)
    expect = np.array(expect)
    observed = np.zeros_like(constraint)
    set_initial_dosage(ploidy, constraint, observed)
    np.testing.assert_array_equal(observed, expect)


def test_initial_dosage__raise_on_ploidy():
    ploidy = 2
    constraint = np.array([1, 0, 0, 0])
    observed = np.zeros_like(constraint)
    with pytest.raises(ValueError, match="Ploidy does not fit within constraint"):
        set_initial_dosage(ploidy, constraint, observed)


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
    "dosage, gamete, expect",
    [
        ([2, 0, 2, 0], [1, 0, 1, 0], [1, 0, 1, 0]),
        ([1, 0, 1, 0], [0, 0, 1, 0], [1, 0, 0, 0]),
        ([1, 2, 1, 0], [1, 1, 0, 0], [0, 1, 1, 0]),
        ([1, 0, 1, 0, 1, 3], [1, 0, 1, 0, 0, 1], [0, 0, 0, 0, 1, 2]),
    ],
)
def test_set_complimentary_gamete(dosage, gamete, expect):
    dosage = np.array(dosage)
    gamete = np.array(gamete)
    expect = np.array(expect)
    compliment = np.zeros_like(dosage)
    set_complimentary_gamete(dosage, gamete, compliment)
    np.testing.assert_array_equal(dosage, gamete + compliment)
    np.testing.assert_array_equal(compliment, expect)


@pytest.mark.parametrize(
    "gamete_dosage, parent_dosage, expect",
    [
        ([1, 1], [1, 1], 0),
        ([1, 1], [2, 0], 0),
        ([2, 0], [1, 0], 1),
        ([2, 0], [2, 0], 2),
    ],
)
def test_double_reduction_permutations(gamete_dosage, parent_dosage, expect):
    gamete_dosage = np.array(gamete_dosage)
    parent_dosage = np.array(parent_dosage)
    observed = double_reduction_permutations(gamete_dosage, parent_dosage)
    assert expect == observed


@pytest.mark.parametrize(
    "parent_dosage, parent_ploidy, gamete_dosage, gamete_ploidy, lambda_, expect",
    [
        ([2, 0], 2, [1, 0], 1, 0.0, 1.0),
        ([1, 1], 2, [1, 0], 1, 0.0, 0.5),
        ([1, 0], 2, [1, 0], 1, 0.0, 0.5),
        ([0, 2], 2, [1, 0], 1, 0.0, 0.0),
        ([1, 1], 2, [1, 1], 2, 0.0, 1.0),  # WGD
        ([0, 2], 2, [0, 2], 2, 0.0, 1.0),  # WGD
        ([1, 1], 2, [1, 1], 2, 0.2, 0.8),  # S/FDR
        ([1, 1], 2, [0, 2], 2, 1.0, 0.5),  # PMR
        ([1, 1], 2, [0, 2], 2, 0.5, 0.25),  # PMR
        ([4, 0, 0, 0], 4, [2, 0, 0, 0], 2, 0.0, 1.0),
        ([0, 0, 4, 0], 4, [0, 0, 2, 0], 2, 0.0, 1.0),
        ([0, 1, 3, 0], 4, [0, 0, 2, 0], 2, 0.0, 0.5),
        ([0, 0, 3, 0], 4, [0, 0, 2, 0], 2, 0.0, 0.5),
        ([0, 2, 2, 0], 4, [0, 1, 1, 0], 2, 0.0, 8 / 12),
        ([0, 2, 0, 1], 4, [0, 1, 1, 0], 2, 0.0, 0.0),
        ([0, 1, 1, 1], 4, [0, 0, 2, 0], 2, 0.0, 0.0),
        ([4, 0, 0, 0], 4, [2, 0, 0, 0], 2, 0.5, 1.0),
        ([1, 1, 1, 1], 4, [2, 0, 0, 0], 2, 0.5, 0.125),
        ([2, 0, 0, 0], 4, [2, 0, 0, 0], 2, 0.5, (2 / 12 + 0.5 * 4 / 12)),
        ([2, 0, 0, 0], 4, [2, 0, 0, 0], 2, 0.1, (2 / 12 + 0.1 * 4 / 12)),
        ([1, 3, 0, 0], 4, [0, 2, 0, 0], 2, 0.5, (6 / 12 + 0.5 * 3 / 12)),
        ([1, 1, 1, 1, 1, 1], 6, [0, 0, 0, 1, 1, 1], 3, 0.0, 6 / 120),
        ([2, 2, 1, 1, 0, 0], 6, [1, 1, 1, 0, 0, 0], 3, 0.0, 24 / 120),
        ([2, 2, 1, 1, 0, 0], 6, [2, 0, 1, 0, 0, 0], 3, 0.0, 6 / 120),
        ([2, 2, 1, 1, 0, 0], 6, [2, 1, 0, 0, 0, 0], 3, 0.0, 12 / 120),
    ],
)
def test_gamete_log_pmf(
    parent_dosage, parent_ploidy, gamete_dosage, gamete_ploidy, lambda_, expect
):
    gamete_dosage = np.array(gamete_dosage)
    parent_dosage = np.array(parent_dosage)
    actual = gamete_log_pmf(
        gamete_dose=gamete_dosage,
        gamete_ploidy=gamete_ploidy,
        parent_dose=parent_dosage,
        parent_ploidy=parent_ploidy,
        gamete_lambda=lambda_,
    )
    np.testing.assert_almost_equal(expect, np.exp(actual))


@pytest.mark.parametrize(
    "seed",
    np.arange(10),
)
def test_gamete_log_pmf__sum_to_one(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    gamete_ploidy = np.random.randint(1, 3)
    parent_ploidy = np.random.randint(2, 4)
    parent_genotype = np.random.randint(n_alleles, size=parent_ploidy)
    n_gametes = comb_with_replacement(n_alleles, gamete_ploidy)
    total_prob = 0.0
    gamete_genotype = np.zeros(gamete_ploidy, int)
    gamete_dosage = np.zeros_like(gamete_genotype)
    parent_dosage = np.zeros_like(gamete_genotype)
    for _ in range(n_gametes):
        set_allelic_dosage(gamete_genotype, gamete_dosage)
        set_parental_copies(parent_genotype, gamete_genotype, parent_dosage)
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
def test_gamete_log_pmf__sum_to_one_lambda(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(1, 10)
    gamete_ploidy = 2
    parent_ploidy = np.random.randint(2, 4)
    parent_genotype = np.random.randint(n_alleles, size=parent_ploidy)
    n_gametes = comb_with_replacement(n_alleles, gamete_ploidy)
    total_prob = 0.0
    lambda_ = np.random.rand()
    gamete_genotype = np.zeros(gamete_ploidy, int)
    gamete_dosage = np.zeros_like(gamete_genotype)
    parent_dosage = np.zeros_like(gamete_genotype)
    for _ in range(n_gametes):
        set_allelic_dosage(gamete_genotype, gamete_dosage)
        set_parental_copies(parent_genotype, gamete_genotype, parent_dosage)
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


def test_gamete_log_pmf__raise_on_non_diploid_lambda():
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
    "parent_count, parent_ploidy, gamete_count, gamete_ploidy, lambda_, expect",
    [
        (2, 2, 1, 1, 0.0, 1.0),
        (1, 2, 1, 1, 0.0, 0.5),
        (0, 2, 1, 1, 0.0, 0.0),
        (1, 2, 1, 2, 0.0, 1.0),  # WGD
        (1, 2, 1, 2, 0.5, 0.5),  # FDR/SDR
        (2, 2, 2, 2, 0.5, 1.0),  # FDR/SDR
        (1, 2, 1, 2, 1.0, 0.0),  # PMR
        (4, 4, 2, 2, 0.0, 1.0),
        (1, 4, 1, 2, 0.0, 1 / 3),
        (2, 4, 1, 2, 0.0, 2 / 3),
        (3, 4, 1, 2, 0.0, 3 / 3),
        (1, 4, 2, 2, 0.0, 0 / 3),
        (2, 4, 2, 2, 0.0, 1 / 3),
        (3, 4, 2, 2, 0.0, 2 / 3),
        (4, 4, 2, 2, 0.0, 3 / 3),
        (0, 4, 2, 2, 0.5, 0.0),  # DR
        (1, 4, 2, 2, 0.5, 0.5),  # DR
        (2, 4, 2, 2, 0.5, (1 / 3 * 0.5 + 0.5)),  # DR
        (2, 4, 2, 2, 0.1, (1 / 3 * 0.9 + 0.1)),  # DR
        (3, 4, 2, 2, 0.1, (2 / 3 * 0.9 + 0.1)),  # DR
        (4, 4, 2, 2, 0.1, 1.0),  # DR
    ],
)
def test_gamete_allele_log_pmf(
    parent_count, parent_ploidy, gamete_count, gamete_ploidy, lambda_, expect
):
    actual = gamete_allele_log_pmf(
        gamete_count=gamete_count,
        gamete_ploidy=gamete_ploidy,
        parent_count=parent_count,
        parent_ploidy=parent_ploidy,
        gamete_lambda=lambda_,
    )
    np.testing.assert_almost_equal(expect, np.exp(actual))


@pytest.mark.parametrize(
    "seed",
    np.arange(20),
)
def test_gamete_allele_log_pmf__sum_to_one(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(15)
    parent_ploidy = np.random.randint(2, 7)
    gamete_ploidy = np.random.randint(1, parent_ploidy)
    parent_genotype = np.random.randint(n_alleles, size=parent_ploidy)
    gamete_genotype = np.random.choice(
        parent_genotype, size=gamete_ploidy, replace=False
    )
    variable_index = np.random.randint(gamete_ploidy)
    total = 0.0
    for i in range(n_alleles):
        gamete_genotype[variable_index] = i
        gamete_count = count_allele(gamete_genotype, i)
        parent_count = count_allele(parent_genotype, i)
        prob = np.exp(
            gamete_allele_log_pmf(
                gamete_count=gamete_count,
                gamete_ploidy=gamete_ploidy,
                parent_count=parent_count,
                parent_ploidy=parent_ploidy,
                gamete_lambda=0.0,
            )
        )
        total += prob
    np.testing.assert_almost_equal(total, 1.0)


@pytest.mark.parametrize(
    "seed",
    np.arange(20),
)
def test_gamete_allele_log_pmf__sum_to_one_lambda(seed):
    np.random.seed(seed)
    n_alleles = np.random.randint(15)
    gamete_lambda = np.random.rand()
    parent_ploidy = np.random.randint(2, 7)
    gamete_ploidy = 2
    parent_genotype = np.random.randint(n_alleles, size=parent_ploidy)
    gamete_genotype = np.random.choice(
        parent_genotype, size=gamete_ploidy, replace=False
    )
    variable_index = np.random.randint(gamete_ploidy)
    total = 0.0
    for i in range(n_alleles):
        gamete_genotype[variable_index] = i
        gamete_count = count_allele(gamete_genotype, i)
        parent_count = count_allele(parent_genotype, i)
        prob = np.exp(
            gamete_allele_log_pmf(
                gamete_count=gamete_count,
                gamete_ploidy=gamete_ploidy,
                parent_count=parent_count,
                parent_ploidy=parent_ploidy,
                gamete_lambda=gamete_lambda,
            )
        )
        total += prob
    np.testing.assert_almost_equal(total, 1.0)


def test_gamete_allele_log_pmf__raise_on_count_greater_than_tau():
    with pytest.raises(AssertionError):
        gamete_allele_log_pmf(
            gamete_count=2,
            gamete_ploidy=1,
            parent_count=2,
            parent_ploidy=2,
            gamete_lambda=0.0,
        )


def test_gamete_allele_log_pmf__raise_on_count_greater_than_ploidy():
    with pytest.raises(AssertionError):
        gamete_allele_log_pmf(
            gamete_count=1,
            gamete_ploidy=1,
            parent_count=3,
            parent_ploidy=2,
            gamete_lambda=0.0,
        )


def test_gamete_allele_log_pmf__raise_on_diploid_lambda():
    with pytest.raises(ValueError):
        gamete_allele_log_pmf(
            gamete_count=1,
            gamete_ploidy=1,
            parent_count=1,
            parent_ploidy=2,
            gamete_lambda=0.1,
        )


def test_gamete_allele_log_pmf__raise_on_hexaploid_lambda():
    with pytest.raises(ValueError):
        gamete_allele_log_pmf(
            gamete_count=1,
            gamete_ploidy=3,
            parent_count=1,
            parent_ploidy=6,
            gamete_lambda=0.1,
        )


@pytest.mark.parametrize(
    "index, parent_dosage, parent_ploidy, gamete_dosage, gamete_ploidy, expect",
    [
        (0, [2, 0], 2, [1, 0], 1, 1.0),  # haploid gamete has no constant
        (0, [1, 1], 2, [1, 0], 1, 1.0),  # haploid gamete has no constant
        (0, [0, 0], 2, [1, 0], 1, 1.0),  # haploid gamete has no constant
        (0, [4, 0, 0, 0], 4, [2, 0, 0, 0], 2, 1.0),
        (0, [2, 2, 0, 0], 4, [2, 0, 0, 0], 2, 2 / 4),
        (0, [1, 1, 0, 0], 4, [1, 1, 0, 0], 2, 0.25),
        (1, [1, 1, 0, 0], 4, [1, 1, 0, 0], 2, 0.25),
        (1, [1, 1, 0, 0], 4, [1, 1, 0, 0], 2, 0.25),
        (
            0,
            [1, 1, 0, 0],
            4,
            [2, 0, 0, 0],
            2,
            0.25,
        ),  # invalid gamete with valid constant
        (1, [1, 1, 1, 0, 0, 0], 6, [1, 1, 1, 0, 0, 0], 3, (2 * 1 / 6 * 1 / 5)),
        (1, [1, 4, 1, 0, 0, 0], 6, [1, 1, 1, 0, 0, 0], 3, (2 * 1 / 6 * 1 / 5)),
        (
            1,
            [2, 1, 1, 0, 0, 0],
            6,
            [1, 1, 1, 0, 0, 0],
            3,
            (2 / 6 * 1 / 5 + 1 / 6 * 2 / 5),
        ),
        (1, [2, 1, 1, 0, 0, 0], 6, [2, 1, 0, 0, 0, 0], 3, (2 / 6 * 1 / 5)),
    ],
)
def test_gamete_const_log_pmf(
    index, parent_dosage, parent_ploidy, gamete_dosage, gamete_ploidy, expect
):
    gamete_dosage = np.array(gamete_dosage)
    parent_dosage = np.array(parent_dosage)
    actual = gamete_const_log_pmf(
        allele_index=index,
        gamete_dose=gamete_dosage,
        gamete_ploidy=gamete_ploidy,
        parent_dose=parent_dosage,
        parent_ploidy=parent_ploidy,
    )
    np.testing.assert_almost_equal(expect, np.exp(actual))


@pytest.mark.parametrize(
    "dosage, frequencies, expect",
    [
        ([4, 0, 0, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 1 / (4**4)),
        ([3, 0, 1, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 4 / (4**4)),
        ([1, 0, 1, 0], [1 / 4, 1 / 4, 1 / 4, 1 / 4], 2 / (4**2)),
        ([1, 0, 1, 0], [0.6, 0.1, 0.1, 0.2], 2 * 0.6 * 0.1),
    ],
)
def test_log_unknown_dosage_prior(dosage, frequencies, expect):
    dosage = np.array(dosage)
    frequencies = np.array(frequencies)
    log_frequencies = np.log(frequencies)
    actual = log_unknown_dosage_prior(dosage, log_frequencies)
    np.testing.assert_almost_equal(np.exp(actual), expect)


@pytest.mark.parametrize(
    "dosage, allele_index, frequencies, expect",
    [
        ([4, 0, 0, 0], 0, [1 / 4, 1 / 4, 1 / 4, 1 / 4], 1 / (4**3)),
        ([4, 0, 0, 0], 1, [1 / 4, 1 / 4, 1 / 4, 1 / 4], 0.0),  # imposable const
        ([3, 0, 1, 0], 0, [1 / 4, 1 / 4, 1 / 4, 1 / 4], 3 / (4**3)),
        ([1, 0, 1, 0], 2, [1 / 4, 1 / 4, 1 / 4, 1 / 4], 1 / 4),
        ([0, 0, 1, 1], 2, [0.6, 0.1, 0.1, 0.2], 0.2),
    ],
)
def test_log_unknown_const_prior(dosage, allele_index, frequencies, expect):
    dosage = np.array(dosage)
    frequencies = np.array(frequencies)
    log_frequencies = np.log(frequencies)
    actual = log_unknown_const_prior(dosage, allele_index, log_frequencies)
    np.testing.assert_almost_equal(np.exp(actual), expect)


def test_trio_log_pmf__sum_to_one__tetraploid():
    np.random.seed(0)
    max_ploidy = 4
    n_alleles = 3
    ploidy_p = 4
    ploidy_q = 4
    parent_p = np.random.randint(n_alleles, size=max_ploidy)
    parent_q = np.random.randint(n_alleles, size=max_ploidy)
    tau_p = 2
    tau_q = 2
    error_p = 0.1
    error_q = 0.1
    frequencies = np.random.rand(n_alleles)
    frequencies /= frequencies.sum()

    # scratch variables
    dosage = np.zeros(max_ploidy, dtype=np.int64)
    dosage_p = np.zeros(max_ploidy, dtype=np.int64)
    dosage_q = np.zeros(max_ploidy, dtype=np.int64)
    gamete_p = np.zeros(max_ploidy, dtype=np.int64)
    gamete_q = np.zeros(max_ploidy, dtype=np.int64)
    constraint_p = np.zeros(max_ploidy, dtype=np.int64)
    constraint_q = np.zeros(max_ploidy, dtype=np.int64)
    dosage_log_frequencies = np.zeros(max_ploidy, dtype=np.float64)

    ploidy = tau_p + tau_q
    n_genotypes = comb_with_replacement(n_alleles, ploidy)

    total_prob = 0.0
    genotype = np.zeros(max_ploidy, dtype=np.int64)
    for _ in range(n_genotypes):
        prob = np.exp(
            trio_log_pmf(
                progeny=genotype,
                parent_p=parent_p,
                parent_q=parent_q,
                ploidy_p=ploidy_p,
                ploidy_q=ploidy_q,
                tau_p=tau_p,
                tau_q=tau_q,
                lambda_p=0.0,
                lambda_q=0.0,
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
                dosage_log_frequencies=dosage_log_frequencies,
            )
        )
        total_prob += prob
        increment_genotype(genotype)
    np.testing.assert_almost_equal(total_prob, 1.0)


@pytest.mark.parametrize(
    "seed",
    np.arange(50),
)
def test_trio_log_pmf__sum_to_one(seed):
    np.random.seed(seed)
    max_ploidy = 7
    n_alleles = np.random.randint(1, 10)
    ploidy_p = np.random.randint(2, max_ploidy)
    ploidy_q = np.random.randint(2, max_ploidy)
    tau_p = np.random.randint(1, ploidy_p)
    tau_q = np.random.randint(1, ploidy_q)

    # adjust max ploidy if exceeded by progeny ploidy
    ploidy = tau_p + tau_q
    max_ploidy = max(max_ploidy, ploidy)

    parent_p = np.random.randint(n_alleles, size=max_ploidy)
    parent_q = np.random.randint(n_alleles, size=max_ploidy)
    parent_p[ploidy_p:] = -2
    parent_q[ploidy_q:] = -2
    error_p = np.random.rand()
    error_q = np.random.rand()
    frequencies = np.random.rand(n_alleles)
    frequencies /= frequencies.sum()

    # scratch variables
    dosage = np.zeros(max_ploidy, dtype=np.int64)
    dosage_p = np.zeros(max_ploidy, dtype=np.int64)
    dosage_q = np.zeros(max_ploidy, dtype=np.int64)
    gamete_p = np.zeros(max_ploidy, dtype=np.int64)
    gamete_q = np.zeros(max_ploidy, dtype=np.int64)
    constraint_p = np.zeros(max_ploidy, dtype=np.int64)
    constraint_q = np.zeros(max_ploidy, dtype=np.int64)
    dosage_log_frequencies = np.zeros(max_ploidy, dtype=np.float64)

    n_genotypes = comb_with_replacement(n_alleles, ploidy)

    total_prob = 0.0
    genotype = np.zeros(max_ploidy, dtype=np.int64)
    genotype[ploidy:] = -2
    for _ in range(n_genotypes):
        prob = np.exp(
            trio_log_pmf(
                progeny=genotype,
                parent_p=parent_p,
                parent_q=parent_q,
                ploidy_p=ploidy_p,
                ploidy_q=ploidy_q,
                tau_p=tau_p,
                tau_q=tau_q,
                lambda_p=0.0,
                lambda_q=0.0,
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
                dosage_log_frequencies=dosage_log_frequencies,
            )
        )
        total_prob += prob
        increment_genotype(genotype[0:ploidy])
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
    max_ploidy = 4
    n_alleles = np.random.randint(1, 10)
    ploidy_p = np.random.randint(2, max_ploidy)
    ploidy_q = np.random.randint(2, max_ploidy)
    parent_p = np.random.randint(n_alleles, size=max_ploidy)
    parent_q = np.random.randint(n_alleles, size=max_ploidy)
    parent_p[ploidy_p:] = -2
    parent_q[ploidy_q:] = -2
    tau_p, tau_q = 2, 2  # progeny ploidy = max_ploidy
    lambda_p = np.random.rand() if use_lambda_p else 0.0
    lambda_q = np.random.rand() if use_lambda_q else 0.0
    error_p = np.random.rand()
    error_q = np.random.rand()
    frequencies = np.random.rand(n_alleles)
    frequencies /= frequencies.sum()

    # scratch variables
    dosage = np.zeros(max_ploidy, dtype=np.int64)
    dosage_p = np.zeros(max_ploidy, dtype=np.int64)
    dosage_q = np.zeros(max_ploidy, dtype=np.int64)
    gamete_p = np.zeros(max_ploidy, dtype=np.int64)
    gamete_q = np.zeros(max_ploidy, dtype=np.int64)
    constraint_p = np.zeros(max_ploidy, dtype=np.int64)
    constraint_q = np.zeros(max_ploidy, dtype=np.int64)
    dosage_log_frequencies = np.zeros(max_ploidy, dtype=np.float64)

    ploidy = tau_p + tau_q
    n_genotypes = comb_with_replacement(n_alleles, ploidy)

    total_prob = 0.0
    genotype = np.zeros(max_ploidy, int)
    genotype[ploidy:] = -2
    for _ in range(n_genotypes):
        prob = np.exp(
            trio_log_pmf(
                progeny=genotype,
                parent_p=parent_p,
                parent_q=parent_q,
                ploidy_p=ploidy_p,
                ploidy_q=ploidy_q,
                tau_p=tau_p,
                tau_q=tau_q,
                lambda_p=lambda_p,
                lambda_q=lambda_q,
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
                dosage_log_frequencies=dosage_log_frequencies,
            )
        )
        total_prob += prob
        increment_genotype(genotype[0:ploidy])
    np.testing.assert_almost_equal(total_prob, 1.0)
