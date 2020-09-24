import pytest
import numpy as np

from mchap import mset
from mchap.assemble import classes


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

    single_chain = mset.repeat(genotypes, counts)

    n_chains = 2
    n_steps = len(single_chain)
    
    genotype_trace = np.tile(single_chain, (n_chains, 1, 1, 1))
    for i in range(n_chains):
        genotype_trace[i] = genotype_trace[i][np.random.permutation(n_steps)]

    llks = np.zeros((n_chains, n_steps))

    trace = classes.GenotypeMultiTrace(genotype_trace, llks)

    burnt = trace.burn(10)
    assert burnt.genotypes.shape == (2, 90, 4, 3)
    assert burnt.llks.shape == (2, 90)

    # posterior sorted from most to least probable
    posterior = trace.posterior()
    np.testing.assert_array_equal(posterior.genotypes, genotypes)
    np.testing.assert_array_equal(posterior.probabilities, counts / counts.sum())
