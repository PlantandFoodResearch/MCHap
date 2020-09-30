import numpy as np
import pytest

from mchap.assemble.mcmc.step import mutation
from mchap.assemble.likelihood import log_likelihood
from mchap.assemble.util import log_likelihoods_as_conditionals
from mchap.assemble.util import seed_numba

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
    mask = np.all(reads == 0.0, axis=0)

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
        genotype_1.tostring(): 0,
        genotype_2.tostring(): 0,
    }
    n_steps = 100_000
    for _ in range(n_steps):
        llk = mutation.base_step(genotype, reads, llk=llk, h=h, j=j, mask=mask[j])
        counts[genotype.tostring()] += 1
    
    actual = np.array([
        counts[genotype_1.tostring()],
        counts[genotype_2.tostring()],
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
        llk = mutation.genotype_compound_step(genotype, reads, llk, mask=mask)
        trace[i] = genotype.copy()

    # count allele 1 occurance
    allele_1_counts = (trace == 1).sum(axis=0).sum(axis=0)
    assert np.all(allele_1_counts < np.array([2000, 15000, 2000]))
    assert np.all(allele_1_counts > np.array([500, 5000, 500]))

    # count allele 2 occurance
    allele_2_counts = (trace == 2).sum(axis=0).sum(axis=0)
    assert np.all(allele_2_counts[mask[:,2]] == 0)
    assert np.all(allele_2_counts[~mask[:,2]] > 0)
