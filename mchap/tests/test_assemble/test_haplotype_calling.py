import numpy as np

from mchap.assemble import haplotype_calling
from mchap.assemble.classes import PosteriorGenotypeDistribution


def test_call_posterior_haplotypes():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 1, 1],  # 1
            [0, 1, 0, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
            [1, 1, 1, 1, 1, 1],  # 4
        ]
    )

    dist1 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[0, 0, 1, 1]],
                haplotypes[[0, 1, 1, 1]],
                haplotypes[[0, 1, 1, 2]],
                haplotypes[[0, 0, 0, 1]],
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.3, 0.1]),
    )
    dist2 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[0, 0, 2, 2]],
                haplotypes[[0, 2, 2, 2]],
                haplotypes[[2, 2, 2, 3]],
                haplotypes[[2, 2, 2, 4]],  # hap 4 at 10%
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.3, 0.1]),
    )
    dist3 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[0, 2, 2, 2]],
                haplotypes[[2, 2, 2, 2]],
                haplotypes[[2, 2, 2, 3]],  # hap 3 at 20%
                haplotypes[[1, 2, 2, 3]],
            ]
        ),
        probabilities=np.array([0.2, 0.6, 0.1, 0.1]),
    )
    posteriors = [dist1, dist2, dist3]
    actual = haplotype_calling.call_posterior_haplotypes(posteriors, threshold=0.15)
    expect = haplotypes[[0, 2, 1, 3]], True  # ref then ordered with called ref
    np.testing.assert_array_equal(actual[0], expect[0])
    assert actual[1] == expect[1]


def test_call_posterior_haplotypes__no_ref():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # 0
            [0, 0, 0, 0, 1, 1],  # 1
            [0, 1, 0, 1, 1, 1],  # 2
            [1, 1, 1, 1, 1, 0],  # 3
            [1, 1, 1, 1, 1, 1],  # 4
        ]
    )

    dist1 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[1, 1, 3, 3]],
                haplotypes[[1, 1, 1, 3]],
                haplotypes[[1, 1, 3, 2]],
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.4]),
    )
    dist2 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[2, 2, 2, 3]],
                haplotypes[[2, 2, 3, 3]],
                haplotypes[[2, 3, 3, 3]],
                haplotypes[[2, 3, 3, 4]],  # hap 4 at 10%
            ]
        ),
        probabilities=np.array([0.2, 0.4, 0.3, 0.1]),
    )
    dist3 = PosteriorGenotypeDistribution(
        genotypes=np.array(
            [
                haplotypes[[1, 2, 2, 2]],
                haplotypes[[2, 2, 2, 2]],
                haplotypes[[2, 2, 2, 3]],
                haplotypes[[1, 2, 2, 3]],
            ]
        ),
        probabilities=np.array([0.2, 0.6, 0.1, 0.1]),
    )
    posteriors = [dist1, dist2, dist3]
    actual = haplotype_calling.call_posterior_haplotypes(posteriors, threshold=0.15)
    expect = haplotypes[[0, 2, 3, 1]], False  # ref not called added at front
    np.testing.assert_array_equal(actual[0], expect[0])
    assert actual[1] == expect[1]
