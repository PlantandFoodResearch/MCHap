import pytest
import numpy as np

from haplokit import mset
from haplokit.assemble import classes


def test_PosteriorGenotypeDistribution():

    genotypes = np.array([
        [[0,0,0],
         [0,0,0],
         [0,0,0],
         [0,0,0]],
        [[0,0,0],
         [0,0,0],
         [0,0,0],
         [1,1,1]],
        [[0,0,0],
         [0,0,0],
         [1,1,1],
         [1,1,1]],
        [[0,0,0],
         [0,0,0],
         [0,0,0],
         [0,1,1]],
    ], dtype=np.int8)
        
    probabilities = np.array([0.1, 0.6, 0.2, 0.1])

    dist = classes.PosteriorGenotypeDistribution(
        genotypes,
        probabilities
    )

    # mode genotype is one with highest probability
    expect_gen, expect_prob = genotypes[1], probabilities[1]
    actual_gen, actual_prob = dist.mode()
    np.testing.assert_array_equal(expect_gen, actual_gen)
    assert expect_prob, actual_prob

    # phenotype is combination of genotypes with same haplotypes
    expect_phen, expect_probs = genotypes[[1, 2]], probabilities[[1,2]]
    actual_phen, actual_probs = dist.mode_phenotype()
    np.testing.assert_array_equal(expect_phen, actual_phen)
    np.testing.assert_array_equal(expect_probs, actual_probs)


def test_GenotypeTrace():

    genotypes = np.array([
        [[0,0,0],
         [0,0,0],
         [0,0,0],
         [1,1,1]],
        [[0,0,0],
         [0,0,0],
         [1,1,1],
         [1,1,1]],
        [[0,0,0],
         [0,0,0],
         [0,0,0],
         [0,0,0]],
        [[0,0,0],
         [0,0,0],
         [0,0,0],
         [0,1,1]],
    ], dtype=np.int8)

    counts = np.array([60, 20, 15, 5])

    genotype_trace = mset.repeat(genotypes, counts)
    genotype_trace = genotype_trace[np.random.permutation(len(genotype_trace))]
    llks = np.zeros(len(genotype_trace))

    trace = classes.GenotypeTrace(genotype_trace, llks)

    burnt = trace.burn(10)
    assert burnt.genotypes.shape == (90, 4, 3)
    assert burnt.llks.shape == (90, )

    # posterior sorted from most to least probable
    posterior = trace.posterior()
    np.testing.assert_array_equal(posterior.genotypes, genotypes)
    np.testing.assert_array_equal(posterior.probabilities, counts / counts.sum())
