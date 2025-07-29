import numpy as np
import pytest

from mchap.assemble import mutation
from mchap.assemble.likelihood import (
    log_likelihood,
    new_log_likelihood_cache,
)
from mchap.assemble.prior import log_genotype_prior
from mchap.jitutils import (
    normalise_log_probs,
    index_as_genotype_alleles,
    seed_numba,
    get_haplotype_dosage,
)
from mchap.encoding import integer
from mchap import combinatorics


@pytest.mark.parametrize(
    "use_cache",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "inbreeding",
    [
        0.0,
        0.25,
        None,  # flat prior
    ],
)
def test_base_step(use_cache, use_read_counts, inbreeding):

    h = 0
    j = 1

    # possible genotypes
    genotype_1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )

    genotype_2 = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )

    # haps 0,1,0 and 0,0,0
    reads = np.array(
        [
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.8, 0.1, 0.1]],
            [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.8, 0.1, 0.1]],
        ]
    )
    log_unique_haplotypes = np.log(2 * 2 * 3)

    # conditional probs of possible genotypes
    llks = np.array(
        [
            log_likelihood(reads, genotype_1),
            log_likelihood(reads, genotype_2),
        ]
    )
    flat_prior = inbreeding is None
    if flat_prior:
        lpriors = np.zeros(2)
    else:
        lpriors = np.array(
            [
                log_genotype_prior(
                    np.array([2, 0]),
                    log_unique_haplotypes=log_unique_haplotypes,
                    inbreeding=inbreeding,
                ),
                log_genotype_prior(
                    np.array([1, 1]),
                    log_unique_haplotypes=log_unique_haplotypes,
                    inbreeding=inbreeding,
                ),
            ]
        )

    # We are only allowing a single haplotype vary so the proposal
    # ratio calculated in `base_step` is biased and this needs to
    # be accounted for in our expected values.
    # In the diploid inbreeding=0 case these values are proportionally
    # inverse to the priors so normalising the llks alone gives the correct
    # result but this is not true if inbreeding > 0.
    log_inverse_proposal_ratio = np.log([2 / 2, 1 / 2])
    expect = normalise_log_probs(llks + lpriors + log_inverse_proposal_ratio)

    # intial genotype
    genotype = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )
    llk = log_likelihood(reads, genotype)

    if use_cache:
        ploidy, n_base = genotype.shape
        _, _, max_alleles = reads.shape
        cache = new_log_likelihood_cache(ploidy, n_base, max_alleles)
    else:
        cache = None

    if use_read_counts:
        reads = reads[[0, 2]]
        read_counts = np.array([2, 2], dtype=int)
    else:
        read_counts = None

    # sample from dist to aproximate conditionals
    seed_numba(42)
    counts = {
        genotype_1.tobytes(): 0,
        genotype_2.tobytes(): 0,
    }
    n_steps = 25_000
    for _ in range(n_steps):
        llk, cache = mutation.base_step(
            genotype,
            reads,
            llk=llk,
            h=h,
            j=j,
            log_unique_haplotypes=log_unique_haplotypes,
            n_alleles=2,
            flat_prior=inbreeding is None,
            inbreeding=inbreeding or 0.0,
            read_counts=read_counts,
            cache=cache,
        )
        counts[genotype.tobytes()] += 1

    actual = (
        np.array(
            [
                counts[genotype_1.tobytes()],
                counts[genotype_2.tobytes()],
            ]
        )
        / n_steps
    )
    assert np.allclose(expect, actual, atol=1e-03)


@pytest.mark.parametrize(
    "use_cache",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "use_read_counts",
    [
        False,
        True,
    ],
)
@pytest.mark.parametrize(
    "inbreeding",
    [
        0.0,
        0.25,
        None,  # flat prior
    ],
)
def test_genotype_compound_step(use_cache, use_read_counts, inbreeding):
    haplotypes = np.array(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=np.int8,
    )
    # reads from haps 0,1,0 and 0,0,0
    reads = np.array(
        [
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],
            [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1]],
            [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
            [[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]],
        ]
    )

    _, n_base, n_nucl = reads.shape
    ploidy = 2
    n_alleles = np.array([n_nucl] * n_base, np.int8)
    unique_haplotypes = combinatorics.count_unique_haplotypes(n_alleles)
    log_unique_haplotypes = np.log(unique_haplotypes)
    assert unique_haplotypes == len(haplotypes)
    unique_genotypes = combinatorics.count_unique_genotypes(unique_haplotypes, ploidy)

    # calculate all possible genotypes
    genotypes = np.zeros((unique_genotypes, ploidy, n_base), dtype=int)
    for i in np.arange(unique_genotypes):
        genotypes[i] = haplotypes[index_as_genotype_alleles(i, ploidy)]

    # calculate expected posterior distribution over all genotypes
    log_expect = np.empty(unique_genotypes, float)
    dosage = np.empty(ploidy, int)
    flat_prior = inbreeding is None
    for i, g in enumerate(genotypes):
        get_haplotype_dosage(dosage, g)
        llk = log_likelihood(reads, g)
        if flat_prior:
            lprior = 0.0
        else:
            lprior = log_genotype_prior(
                dosage,
                log_unique_haplotypes=log_unique_haplotypes,
                inbreeding=inbreeding,
            )
        log_expect[i] = llk + lprior
    expect = normalise_log_probs(log_expect)

    # additional parameters
    if use_cache:
        cache = new_log_likelihood_cache(ploidy, n_base, n_nucl)
    else:
        cache = None
    if use_read_counts:
        reads = reads[[0, 2]]
        read_counts = np.array([2, 2])
    else:
        read_counts = None

    # mcmc simulation
    genotype = genotypes[0].copy()
    llk = log_likelihood(reads, genotype)

    # count occurrence of each genotype in MCMC
    counts = {g.tobytes(): 0 for g in genotypes}
    for _ in range(25_000):
        llk, cache = mutation.compound_step(
            genotype,
            reads,
            llk,
            n_alleles=n_alleles,
            log_unique_haplotypes=log_unique_haplotypes,
            flat_prior=inbreeding is None,
            inbreeding=inbreeding or 0.0,
            read_counts=read_counts,
            cache=cache,
        )
        genotype = integer.sort(genotype)
        counts[genotype.tobytes()] += 1

    counts_array = np.array([counts[g.tobytes()] for g in genotypes])
    actual = counts_array / counts_array.sum()

    assert np.allclose(expect, actual, atol=1e-02)


@pytest.mark.parametrize(
    "flat_prior",
    [
        False,
        True,
    ],
)
def test_genotype_compound_step__mask_ragged(flat_prior):

    # haps 0,1,0 and 0,0,0
    reads = np.array(
        [
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            [[0.9, 0.1, 0.0], [0.1, 0.9, 0.0], [0.8, 0.1, 0.1]],
            [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.8, 0.1, 0.1]],
            [[0.9, 0.1, 0.0], [0.9, 0.1, 0.0], [0.8, 0.1, 0.1]],
        ]
    )
    mask = np.all(reads == 0.0, axis=0)
    n_alleles = np.sum(~mask, axis=-1)
    log_unique_haplotypes = np.sum(np.log(n_alleles))

    # initial genotype
    genotype = np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=np.int8,
    )
    llk = log_likelihood(reads, genotype)

    n_steps = 10_000
    ploidy, n_base = genotype.shape
    trace = np.zeros((n_steps, ploidy, n_base), dtype=np.int8) - 1

    seed_numba(42)
    for i in range(n_steps):
        llk, _ = mutation.compound_step(
            genotype,
            reads,
            llk,
            n_alleles=n_alleles,
            log_unique_haplotypes=log_unique_haplotypes,
            flat_prior=flat_prior,
            cache=None,
        )
        trace[i] = genotype.copy()

    # count snp allele 1 occurrence in each base position
    allele_1_counts = (trace == 1).sum(axis=0).sum(axis=0)
    assert np.all(allele_1_counts < np.array([2000, 15000, 2000]))
    assert np.all(allele_1_counts > np.array([500, 5000, 500]))

    # check snp allele 2 occurs only in base position 3
    allele_2_counts = (trace == 2).sum(axis=0).sum(axis=0)
    assert np.all(allele_2_counts[0:2] == 0)
    assert allele_2_counts[2] > 0
