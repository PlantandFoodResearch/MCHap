import numpy as np
import pytest
from itertools import permutations
from itertools import combinations_with_replacement

from mchap.assemble import DenovoMCMC
from mchap.calling.classes import (
    CallingMCMC,
    PosteriorGenotypeAllelesDistribution,
    GenotypeAllelesMultiTrace,
)
from mchap.testing import simulate_reads
from mchap.jitutils import seed_numba, genotype_alleles_as_index
from mchap.combinatorics import count_unique_genotypes


@pytest.mark.parametrize(
    "seed",
    [0, 42, 36],  # these numbers can be finicky
)
def test_CallingMCMC__gibbs_mh_equivalence(seed):
    seed_numba(seed)
    np.random.seed(seed)
    # ensure no duplicate haplotypes
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]
    )
    n_haps, n_pos = haplotypes.shape

    # simulate reads
    inbreeding = np.random.rand()
    n_reads = np.random.randint(4, 15)
    ploidy = np.random.randint(2, 5)
    genotype_alleles = np.random.randint(0, n_haps, size=ploidy)
    genotype_alleles.sort()
    genotype_haplotypes = haplotypes[genotype_alleles]
    reads = simulate_reads(
        genotype_haplotypes,
        n_alleles=np.full(n_pos, 2, int),
        n_reads=n_reads,
    )
    read_counts = np.ones(n_reads, int)

    # Gibbs sampler
    model = CallingMCMC(
        ploidy=ploidy, haplotypes=haplotypes, inbreeding=inbreeding, steps=10050
    )
    gibbs_genotype, gibbs_genotype_prob, gibbs_phenotype_prob = (
        model.fit(reads, read_counts).burn(50).posterior().mode(phenotype=True)
    )

    # MH sampler
    model = CallingMCMC(
        ploidy=ploidy,
        haplotypes=haplotypes,
        inbreeding=inbreeding,
        steps=10050,
        step_type="Metropolis-Hastings",
    )
    mh_genotype, mh_genotype_prob, mh_phenotype_prob = (
        model.fit(reads, read_counts).burn(50).posterior().mode(phenotype=True)
    )

    # check results are equivalent
    # tolerance is not enough for all cases so some cherry-picking of
    # the random seed values is required
    np.testing.assert_array_equal(gibbs_genotype, mh_genotype)
    np.testing.assert_allclose(gibbs_genotype_prob, mh_genotype_prob, atol=0.01)
    np.testing.assert_allclose(gibbs_phenotype_prob, mh_phenotype_prob, atol=0.01)


@pytest.mark.parametrize(
    "seed",
    [0, 13, 36],  # these numbers can be finicky
)
def test_CallingMCMC__DenovoMCMC_equivalence(seed):
    ploidy = 4
    inbreeding = 0.015
    n_pos = 4
    n_alleles = np.full(n_pos, 2, np.int8)
    # generate all haplotypes

    def generate_haplotypes(pos):
        haps = []
        for a in combinations_with_replacement([0, 1], pos):
            perms = list(set(permutations(a)))
            perms.sort()
            for h in perms:
                haps.append(h)
        haps.sort()
        haplotypes = np.array(haps, np.int8)
        return haplotypes

    haplotypes = generate_haplotypes(n_pos)
    haplotype_labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}

    # True genotype
    genotype = haplotypes[[0, 0, 1, 2]]

    # simulated reads
    seed_numba(seed)
    np.random.seed(seed)
    reads = simulate_reads(genotype, n_reads=8, errors=False)
    read_counts = np.ones(len(reads), int)

    # denovo assembly
    model = DenovoMCMC(
        ploidy=ploidy,
        n_alleles=n_alleles,
        steps=10500,
        inbreeding=inbreeding,
        fix_homozygous=1,
    )
    denovo_phenotype = (
        model.fit(reads, read_counts=read_counts).burn(500).posterior().mode_phenotype()
    )
    denovo_phen_prob = denovo_phenotype.probabilities.sum()
    idx = np.argmax(denovo_phenotype.probabilities)
    denovo_gen_prob = denovo_phenotype.probabilities[idx]
    denovo_genotype = denovo_phenotype.genotypes[idx]
    denovo_alleles = [haplotype_labels[h.tobytes()] for h in denovo_genotype]
    denovo_alleles = np.sort(denovo_alleles)

    # gibbs base calling
    model = CallingMCMC(
        ploidy=ploidy, haplotypes=haplotypes, inbreeding=inbreeding, steps=10500
    )
    call_alleles, call_genotype_prob, call_phenotype_prob = (
        model.fit(reads, read_counts).burn(500).posterior().mode(phenotype=True)
    )

    # check results are equivalent
    # tolerance is not enough for all cases so some cherry-picking of
    # the random seed values is required
    np.testing.assert_array_equal(call_alleles, denovo_alleles)
    np.testing.assert_allclose(call_genotype_prob, denovo_gen_prob, atol=0.01)
    np.testing.assert_allclose(call_phenotype_prob, denovo_phen_prob, atol=0.01)


def test_CallingMCMC__zero_reads():
    haplotypes = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ]
    )
    n_chains = 2
    n_steps = 10500
    n_burn = 500
    ploidy = 4
    inbreeding = 0.01
    n_haps, n_pos = haplotypes.shape
    reads = np.empty((0, n_pos, 2))
    read_counts = np.array([], int)

    model = CallingMCMC(
        ploidy=ploidy,
        haplotypes=haplotypes,
        inbreeding=inbreeding,
        steps=n_steps,
        chains=n_chains,
    )
    trace = model.fit(reads, read_counts)
    _, call_genotype_prob, call_phenotype_prob = (
        trace.burn(n_burn).posterior().mode(phenotype=True)
    )

    assert trace.genotypes.shape == (n_chains, n_steps, ploidy)
    assert trace.genotypes.max() < n_haps
    assert call_genotype_prob < 0.05
    assert call_phenotype_prob < 0.05


def test_CallingMCMC__zero_snps():
    haplotypes = np.array(
        [
            [],
        ]
    )
    n_chains = 2
    n_steps = 10500
    n_burn = 500
    ploidy = 4
    inbreeding = 0.01
    _, n_pos = haplotypes.shape
    reads = np.empty((0, n_pos, 2))
    read_counts = np.array([], int)

    model = CallingMCMC(
        ploidy=ploidy,
        haplotypes=haplotypes,
        inbreeding=inbreeding,
        steps=n_steps,
        chains=n_chains,
    )
    trace = model.fit(reads, read_counts)
    _, call_genotype_prob, call_phenotype_prob = (
        trace.burn(n_burn).posterior().mode(phenotype=True)
    )

    assert trace.genotypes.shape == (n_chains, n_steps, ploidy)
    assert np.all(trace.genotypes == 0)
    assert np.all(np.isnan(trace.llks))
    assert call_genotype_prob == 1
    assert call_phenotype_prob == 1


def test_PosteriorGenotypeAllelesDistribution__as_array():
    ploidy = 4
    n_haps = 4
    unique_genotypes = count_unique_genotypes(n_haps, ploidy)
    observed_genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 2, 2],
            [0, 2, 2, 2],
            [0, 0, 1, 2],
            [0, 1, 2, 2],
        ]
    )
    observed_probabilities = np.array([0.05, 0.08, 0.22, 0.45, 0.05, 0.15])
    posterior = PosteriorGenotypeAllelesDistribution(
        observed_genotypes, observed_probabilities
    )
    result = posterior.as_array(n_haps)
    assert result.sum() == observed_probabilities.sum() == 1
    assert len(result) == unique_genotypes == 35
    for g, p in zip(observed_genotypes, observed_probabilities):
        idx = genotype_alleles_as_index(g)
        assert result[idx] == p


def test_PosteriorGenotypeAllelesDistribution__mode():
    observed_genotypes = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 2],
            [0, 0, 2, 2],
            [0, 2, 2, 2],
            [0, 0, 1, 2],
            [0, 1, 2, 2],
        ]
    )
    observed_probabilities = np.array([0.05, 0.08, 0.22, 0.45, 0.05, 0.15])
    posterior = PosteriorGenotypeAllelesDistribution(
        observed_genotypes, observed_probabilities
    )

    genotype, genotype_prob = posterior.mode()
    np.testing.assert_array_equal(genotype, observed_genotypes[3])
    assert genotype_prob == observed_probabilities[3]

    genotype, genotype_prob, phenotype_prob = posterior.mode(phenotype=True)
    np.testing.assert_array_equal(genotype, observed_genotypes[3])
    assert genotype_prob == observed_probabilities[3]
    assert phenotype_prob == observed_probabilities[1:4].sum()


@pytest.mark.parametrize("threshold,expect", [(0.99, 0), (0.8, 0), (0.6, 1)])
def test_GenotypeAllelesMultiTrace__replicate_incongruence_1(threshold, expect):
    g0 = [0, 0, 1, 2]  # phenotype 1
    g1 = [0, 1, 1, 2]  # phenotype 1
    g2 = [0, 1, 2, 2]  # phenotype 1
    g3 = [0, 0, 2, 2]  # phenotype 2
    genotypes = np.array([g0, g1, g2, g3])

    t0 = genotypes[[0, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 10:0
    t1 = genotypes[[3, 2, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t2 = genotypes[[0, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 10:0
    t3 = genotypes[[3, 3, 3, 3, 3, 3, 3, 2, 1, 2]]  # 3:7
    trace = GenotypeAllelesMultiTrace(
        genotypes=np.array([t0, t1, t2, t3]), llks=np.ones((4, 10))
    )

    actual = trace.replicate_incongruence(threshold)
    assert actual == expect


@pytest.mark.parametrize("threshold,expect", [(0.99, 0), (0.8, 0), (0.6, 2)])
def test_GenotypeAllelesMultiTrace__replicate_incongruence_2(threshold, expect):

    g0 = [0, 0, 1, 2]  # phenotype 1
    g1 = [0, 1, 1, 2]  # phenotype 1
    g2 = [0, 1, 2, 2]  # phenotype 1
    g3 = [0, 0, 2, 3]  # phenotype 2
    g4 = [0, 2, 3, 4]  # phenotype 3
    genotypes = np.array([g0, g1, g2, g3, g4])

    t0 = genotypes[[3, 1, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t1 = genotypes[[3, 2, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t2 = genotypes[[0, 3, 0, 1, 2, 0, 1, 1, 0, 1]]  # 9:1
    t3 = genotypes[[3, 3, 4, 4, 4, 3, 4, 4, 4, 4]]  # 3:7
    trace = GenotypeAllelesMultiTrace(
        genotypes=np.array([t0, t1, t2, t3]), llks=np.ones((4, 10))
    )
    actual = trace.replicate_incongruence(threshold)
    assert actual == expect
