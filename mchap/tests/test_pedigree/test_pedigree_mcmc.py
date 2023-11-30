import pytest
import numpy as np

from mchap.testing import simulate_reads
from mchap.pedigree.mcmc import (
    metropolis_hastings_probabilities,
    gibbs_probabilities,
    sample_children_matrix,
)


SINGLETON_PEDIGREE = {
    "parent": [
        [-1, -1],
    ],
    "tau": [
        [1, 1],
    ],
    "lambda": [
        [0, 0],
    ],
    "genotype": [
        [0, 0],
    ],
}


SINGLETON_CLONE_PEDIGREE = {
    "parent": [[-1, -1]],
    "tau": [[2, 0]],
    "lambda": [
        [0.0, 0.0],
    ],
    "genotype": [
        [0, 2],
    ],
}


DIPLOID_TRIO_PEDIGREE = {
    "parent": [
        [-1, -1],
        [-1, -1],
        [0, 1],
    ],
    "tau": [
        [1, 1],
        [1, 1],
        [1, 1],
    ],
    "lambda": [
        [0, 0],
        [0, 0],
        [0, 0],
    ],
    "genotype": [
        [0, 0],
        [1, 2],
        [0, 2],
    ],
}


TETRAPLOID_DUO_PEDIGREE = {
    "parent": [
        [-1, -1],
        [0, -1],  # unknown parent
    ],
    "tau": [
        [2, 2],
        [2, 2],
    ],
    "lambda": [
        [0.1, 0.1],
        [0.1, 0.1],
    ],
    "genotype": [
        [1, 1, 2, 3],
        [1, 1, 2, 1],
    ],
}


TETRAPLOID_DUO_PEDIGREE_INCONGRUENT = TETRAPLOID_DUO_PEDIGREE.copy()
TETRAPLOID_DUO_PEDIGREE_INCONGRUENT["genotype"] = [
    [1, 1, 2, 3],
    [0, 0, 2, 4],  # incongruent progeny
]


TETRAPLOID_TRIO_PEDIGREE = {
    "parent": [
        [-1, -1],
        [-1, -1],
        [0, 1],
    ],
    "tau": [
        [2, 2],
        [2, 2],
        [2, 2],
    ],
    "lambda": [
        [0.1, 0.1],
        [0.1, 0.1],
        [0.1, 0.1],
    ],
    "genotype": [
        [0, 0, 1, 2],
        [1, 1, 2, 3],
        [1, 1, 2, 1],
    ],
}


TETRAPLOID_TRIO_PEDIGREE_INCONGRUENT = TETRAPLOID_TRIO_PEDIGREE.copy()
TETRAPLOID_TRIO_PEDIGREE_INCONGRUENT["genotype"] = [
    [0, 0, 1, 2],
    [1, 1, 2, 3],
    [0, 0, 0, 4],  # incongruent progeny
]


UNBALANCED_TRIO_PEDIGREE = {
    "parent": [
        [-1, -1],
        [-1, -1],
        [0, 1],
    ],
    "tau": [
        [1, 1],
        [3, 3],
        [1, 3],
    ],
    "lambda": [
        [0, 0],
        [0, 0],
        [0, 0],
    ],
    "genotype": [
        [0, 1, -2, -2, -2, -2],
        [1, 1, 2, 2, 3, 4],
        [1, 1, 2, 4, -2, -2],
    ],
}


MIXED_QUARTET_PEDIGREE = {
    "parent": [
        [-1, -1],
        [-1, -1],
        [0, 1],
        [-1, 2],  # unknown parent
    ],
    "tau": [
        [1, 1],
        [2, 2],
        [2, 2],
        [2, 2],
    ],
    "lambda": [
        [0.0, 0.0],
        [0.01, 0.01],
        [0.345, 0.01],
        [0.01, 0.01],
    ],
    "genotype": [
        [0, 1, -2, -2],
        [0, 0, 0, 2],
        [0, 1, 1, 2],
        [0, 0, 1, 2],
    ],
}


HAMILTON_KERR_PEDIGREE = {
    "parent": [[-1, -1], [-1, -1], [-1, 1], [0, -1], [0, 2], [0, 2], [5, 1], [5, 1]],
    "tau": [[1, 1], [2, 2], [0, 2], [2, 0], [1, 1], [2, 2], [2, 2], [2, 2]],
    "lambda": [
        [0, 0],
        [0.167, 0.167],
        [0, 0.167],
        [0.041, 0],
        [0, 0],
        [0.918, 0.041],
        [0.167, 0.167],
        [0.167, 0.167],
    ],
    "genotype": [
        [0, 1, -2, -2],
        [0, 2, 2, 3],
        [0, 2, -2, -2],
        [0, 1, -2, -2],
        [1, 2, -2, -2],
        [0, 0, 0, 2],
        [0, 0, 2, 2],
        [0, 2, 2, 3],
    ],
}


# The same pedigree but with some incongruent genotypes
HAMILTON_KERR_PEDIGREE_INCONGRUENT = HAMILTON_KERR_PEDIGREE.copy()
HAMILTON_KERR_PEDIGREE_INCONGRUENT["genotype"] = [
    [0, 1, -2, -2],
    [0, 2, 2, 3],
    [0, 2, -2, -2],
    [0, 1, -2, -2],
    [1, 1, -2, -2],  # only one parent has a '1'
    [0, 0, 0, 2],
    [0, 0, 2, 5],  # novel allele
    [0, 2, 2, 3],
]


@pytest.mark.parametrize("gamete_error", [0.001, 0.1, 0.5, 1.0, "random"])
@pytest.mark.parametrize("read_depth", [4, 8])
@pytest.mark.parametrize(
    "pedigree",
    [
        SINGLETON_PEDIGREE,
        SINGLETON_CLONE_PEDIGREE,
        DIPLOID_TRIO_PEDIGREE,
        TETRAPLOID_DUO_PEDIGREE,
        TETRAPLOID_DUO_PEDIGREE_INCONGRUENT,
        TETRAPLOID_TRIO_PEDIGREE,
        TETRAPLOID_TRIO_PEDIGREE_INCONGRUENT,
        UNBALANCED_TRIO_PEDIGREE,
        MIXED_QUARTET_PEDIGREE,
        HAMILTON_KERR_PEDIGREE,
        HAMILTON_KERR_PEDIGREE_INCONGRUENT,
    ],
)
def test_gibbs_mh_probabilities_equivalence(pedigree, read_depth, gamete_error):

    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0, 1],
            [0, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 1, 1],
        ]
    )
    n_alleles, n_pos = haplotypes.shape
    np.random.seed(0)
    frequencies = np.random.rand(n_alleles)
    print("Freqs:", frequencies)

    # pedigree and genotype data
    genotypes = np.array(pedigree["genotype"])
    sample_parents = np.array(pedigree["parent"], int)
    sample_children = sample_children_matrix(sample_parents)
    print(sample_children)
    gamete_tau = np.array(pedigree["tau"], int)
    gamete_lambda = np.array(pedigree["lambda"], float)
    n_samples, max_ploidy = genotypes.shape
    assert sample_parents.shape == (n_samples, 2)
    assert gamete_tau.shape == (n_samples, 2)
    assert gamete_lambda.shape == (n_samples, 2)
    if gamete_error == "random":
        gamete_error = np.random.rand(n_samples * 2).reshape(n_samples, 2)
    else:
        gamete_error = np.full((n_samples, 2), gamete_error, float)
    sample_ploidy = gamete_tau.sum(axis=-1)

    # simulate reads from genotypes
    sample_read_dists = np.empty((n_samples, read_depth, n_pos, 2))
    np.random.seed(0)
    for i, g in enumerate(genotypes):
        sample_read_dists[i] = simulate_reads(
            haplotypes[g],
            n_alleles=2,
            n_reads=read_depth,
            errors=False,
            uniform_sample=False,
        )
    sample_read_counts = np.ones((n_samples, read_depth), dtype=int)

    # scratch variables
    dosage = np.zeros(max_ploidy, dtype=np.int64)
    dosage_p = np.zeros(max_ploidy, dtype=np.int64)
    dosage_q = np.zeros(max_ploidy, dtype=np.int64)
    gamete_p = np.zeros(max_ploidy, dtype=np.int64)
    gamete_q = np.zeros(max_ploidy, dtype=np.int64)
    constraint_p = np.zeros(max_ploidy, dtype=np.int64)
    constraint_q = np.zeros(max_ploidy, dtype=np.int64)

    # test over all alleles of all samples
    for target_index in range(n_samples):
        for allele_index in range(sample_ploidy[target_index]):
            print(target_index, allele_index)

            # gibbs probs
            gibbs = gibbs_probabilities(
                target_index,
                allele_index,
                genotypes,
                sample_ploidy,
                sample_parents,
                sample_children,
                gamete_tau,
                gamete_lambda,
                gamete_error,
                sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
                sample_read_counts,  # array (n_samples, n_reads)
                haplotypes,  # (n_haplotypes, n_pos)
                log_frequencies=np.log(frequencies),
                llk_cache=None,
                dosage=dosage,
                dosage_p=dosage_p,
                dosage_q=dosage_q,
                gamete_p=gamete_p,
                gamete_q=gamete_q,
                constraint_p=constraint_p,
                constraint_q=constraint_q,
            )

            # MH probs
            mtx = []
            for i in range(n_alleles):
                genotypes[target_index, allele_index] = i
                probs = metropolis_hastings_probabilities(
                    target_index,
                    allele_index,
                    genotypes,
                    sample_ploidy,
                    sample_parents,
                    sample_children,
                    gamete_tau,
                    gamete_lambda,
                    gamete_error,
                    sample_read_dists,  # array (n_samples, n_reads, n_pos, n_nucl)
                    sample_read_counts,  # array (n_samples, n_reads)
                    haplotypes,  # (n_haplotypes, n_pos)
                    log_frequencies=np.log(frequencies),
                    llk_cache=None,
                    dosage=dosage,
                    dosage_p=dosage_p,
                    dosage_q=dosage_q,
                    gamete_p=gamete_p,
                    gamete_q=gamete_q,
                    constraint_p=constraint_p,
                    constraint_q=constraint_q,
                )
                mtx.append(probs)
            mtx = np.array(mtx)
            longrun = np.linalg.matrix_power(mtx, 100)[0]
            assert not np.isnan(gibbs).any()
            np.testing.assert_almost_equal(longrun, gibbs)


@pytest.mark.parametrize(
    "parent, children",
    [
        (
            [
                [-1, -1],
                [-1, -1],
                [0, 1],
                [-1, 2],
            ],
            [
                [2],
                [2],
                [3],
                [-1],
            ],
        ),
        (
            [
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [-1, -1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [0, 1],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [2, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
                [0, 3],
            ],
            [
                [4, 5, 6, 7, 8, 15, 16, 17, 18, 19, -1],
                [4, 5, 6, 7, 8, -1, -1, -1, -1, -1, -1],
                [9, 10, 11, 12, 13, 14, -1, -1, -1, -1, -1],
                [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            ],
        ),
    ],
)
def test_sample_children_matrix(parent, children):
    parent = np.array(parent)
    children = np.array(children)
    np.testing.assert_array_equal(sample_children_matrix(parent), children)
