import numpy as np
import pytest

from mchap.assemble.mcmc.step import mutation
from mchap.assemble.likelihood import log_likelihood
from mchap.assemble.util import log_likelihoods_as_conditionals
from mchap.assemble.util import seed_numba
from mchap.encoding import integer


def test_base_step():

    h = 0
    j = 1

    # possible genotypes
    genotype_1 = np.array([
        [0, 0, 0],
        [0, 0, 0],
    ], dtype=np.int8)

    genotype_2 = np.array([
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.int8)

    # haps 0,1,0 and 0,0,0
    reads = np.array([
        [[0.9, 0.1, 0.0],
         [0.1, 0.9, 0.0],
         [0.8, 0.1, 0.1]],
        [[0.9, 0.1, 0.0],
         [0.1, 0.9, 0.0],
         [0.8, 0.1, 0.1]],
        [[0.9, 0.1, 0.0],
         [0.9, 0.1, 0.0],
         [0.8, 0.1, 0.1]],
        [[0.9, 0.1, 0.0],
         [0.9, 0.1, 0.0],
         [0.8, 0.1, 0.1]],
    ])
    u_haps = int(2 * 2 * 3)

    # conditional probs of possible genotypes
    llks = np.array([
        log_likelihood(reads, genotype_1),
        log_likelihood(reads, genotype_2),
    ])
    expect = log_likelihoods_as_conditionals(llks)

    # intial genotype
    genotype = np.array([
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    llk = log_likelihood(reads, genotype)
    
    # sample from dist to aproximate conditionals
    seed_numba(42)
    counts = {
        genotype_1.tobytes(): 0,
        genotype_2.tobytes(): 0,
    }
    n_steps = 100_000
    for _ in range(n_steps):
        llk = mutation.base_step(
            genotype,
            reads,
            llk=llk,
            h=h,
            j=j,
            unique_haplotypes=u_haps,
            n_alleles=2,
        )
        counts[genotype.tobytes()] += 1
    
    actual = np.array([
        counts[genotype_1.tobytes()],
        counts[genotype_2.tobytes()],
    ]) / n_steps

    assert np.allclose(expect, actual, atol=1e-03)


def test_genotype_compound_step():

    # haps 0,1,0 and 0,0,0
    reads = np.array([
        [[0.9, 0.1, 0.0],
         [0.1, 0.9, 0.0],
         [0.8, 0.1, 0.1]],
        [[0.9, 0.1, 0.0],
         [0.1, 0.9, 0.0],
         [0.8, 0.1, 0.1]],
        [[0.9, 0.1, 0.0],
         [0.9, 0.1, 0.0],
         [0.8, 0.1, 0.1]],
        [[0.9, 0.1, 0.0],
         [0.9, 0.1, 0.0],
         [0.8, 0.1, 0.1]],
    ])
    mask = np.all(reads == 0.0, axis=0)
    n_alleles = np.sum(~mask, axis=-1).astype(np.int8)

    # intial genotype
    genotype = np.array([
        [0, 1, 0],
        [0, 0, 0],
    ], dtype=np.int8)
    llk = log_likelihood(reads, genotype)

    n_steps = 10_000
    ploidy, n_base = genotype.shape
    trace = np.zeros((n_steps, ploidy, n_base), dtype=np.int8) -1

    seed_numba(42)
    for i in range(n_steps):
        llk = mutation.compound_step(genotype, reads, llk, n_alleles=n_alleles)
        trace[i] = genotype.copy()

    # count allele 1 occurance
    allele_1_counts = (trace == 1).sum(axis=0).sum(axis=0)
    assert np.all(allele_1_counts < np.array([2000, 15000, 2000]))
    assert np.all(allele_1_counts > np.array([500, 5000, 500]))

    # count allele 2 occurance
    allele_2_counts = (trace == 2).sum(axis=0).sum(axis=0)
    assert np.all(allele_2_counts[mask[:,2]] == 0)
    assert np.all(allele_2_counts[~mask[:,2]] > 0)


def test_genotype_compound_step__posterior():
    # haps 0,1,0 and 0,0,0
    reads = np.array([
        [[0.9, 0.1],
         [0.1, 0.9],
         [0.9, 0.1]],
        [[0.9, 0.1],
         [0.1, 0.9],
         [0.9, 0.1]],
        [[0.9, 0.1],
         [0.9, 0.1],
         [0.9, 0.1]],
        [[0.9, 0.1],
         [0.9, 0.1],
         [0.9, 0.1]],
    ])
    mask = np.all(reads == 0.0, axis=0)
    n_alleles = np.sum(~mask, axis=-1).astype(np.int8)

    genotypes = np.array([
        [[0, 0],  # 2
         [0, 0]],
        [[0, 0],  # 1:1
         [0, 1]],
        [[0, 0],  # 1:1
         [1, 0]],
        [[0, 0],  # 1:1
         [1, 1]],
        [[0, 1],  # 2
         [0, 1]],
        [[0, 1],  # 1:1
         [1, 0]],
        [[0, 1],  # 1:1
         [1, 1]],
        [[1, 0],  # 2
         [1, 0]],
        [[1, 0],  # 1:1
         [1, 1]],
        [[1, 1],  # 2
         [1, 1]],
    ], dtype=np.int8)

    # llk of each genotype
    llks = np.array([log_likelihood(reads, g) for g in genotypes])

    # prior probability of each genotype based on dosage
    priors = np.array([1, 2, 2, 2, 1, 2, 2, 1, 2, 1])
    priors = priors / priors.sum()

    # posterior probabilities from priors and likelihoods
    exact_posteriors = np.exp(llks + np.log(priors))
    exact_posteriors = exact_posteriors / exact_posteriors.sum()


    # now run MCMC simulation
    # initial genotype
    genotype = np.array([
        [0, 0],
        [0, 0],
    ], dtype=np.int8)
    llk = log_likelihood(reads, genotype)
    # count choices of each option
    counts = {}
    for g in genotypes:
        counts[g.tobytes()] = 0
    # simulation
    for _ in range(100000):
        llk = mutation.compound_step(
            genotype, 
            reads, 
            llk, 
            n_alleles=n_alleles
        )
        genotype = integer.sort(genotype)
        counts[genotype.tobytes()] += 1
    totals = np.zeros(len(genotypes), dtype=int)
    for i, g in enumerate(genotypes):
        totals[i] = counts[g.tobytes()]
    
    simulation_posteriors = totals / totals.sum()

    np.testing.assert_array_almost_equal(
        exact_posteriors,
        simulation_posteriors,
        decimal=3,
    )
