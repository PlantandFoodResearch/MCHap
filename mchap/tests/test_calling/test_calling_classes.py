import numpy as np
import pytest
from itertools import permutations
from itertools import combinations_with_replacement

from mchap.assemble import DenovoMCMC
from mchap.calling.classes import CallingMCMC
from mchap.testing import simulate_reads
from mchap.assemble.util import seed_numba


@pytest.mark.parametrize(
    "seed",
    [0, 42, 36],  # these numbers can be finicky
)
def test_mcmc_gibbs_mh_equivalence(seed):
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
def test_mcmc_calling_assemble_equivalence(seed):
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
