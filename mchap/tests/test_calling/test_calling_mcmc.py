import numpy as np
import pytest

from mchap.testing import simulate_reads
from mchap.calling.mcmc import gibbs_options, mh_options


@pytest.mark.parametrize(
    "seed",
    [11, 42, 13, 0, 12234, 213, 45436, 1312, 374645],
)
def test_gibbs_mh_transition_equivalence(seed):
    """Tests that transition matrices of gibbs and
    metropolis-hastings methods are equivalent.
    """
    # simulate haplotypes and reads
    np.random.seed(seed)
    inbreeding = np.random.rand()
    n_pos = np.random.randint(3, 13)
    n_haps = np.random.randint(2, 20)
    n_reads = np.random.randint(2, 15)
    ploidy = np.random.randint(2, 9)
    haplotypes = np.random.randint(0, 2, size=(n_haps, n_pos))
    n_alleles = len(haplotypes)
    genotype_alleles = np.random.randint(0, n_haps, size=ploidy)
    genotype_alleles.sort()
    genotype_haplotypes = haplotypes[genotype_alleles]
    reads = simulate_reads(
        genotype_haplotypes,
        n_alleles=np.full(n_pos, 2, int),
        n_reads=n_reads,
    )
    read_counts = np.random.randint(1, 10, size=n_reads)

    # single allele to vary
    variable_allele = np.random.randint(0, ploidy)

    # gibbs transition probabilities for the variable allele
    # these should be equivalent to the long run behavior of
    # the MH algorithm
    gibbs_llks_array = np.zeros(n_alleles)
    gibbs_lpriors_array = np.zeros(n_alleles)
    gibbs_probabilities_array = np.zeros(n_alleles)
    gibbs_options(
        genotype_alleles=genotype_alleles,
        variable_allele=variable_allele,
        haplotypes=haplotypes,
        reads=reads,
        read_counts=read_counts,
        inbreeding=inbreeding,
        llks_array=gibbs_llks_array,
        lpriors_array=gibbs_lpriors_array,
        probabilities_array=gibbs_probabilities_array,
    )
    gibbs_long_run = gibbs_probabilities_array.copy()

    # calculate the long run behavior of the MH approach
    # by evaluating the transition probabilities for each
    # possible initial allele
    mh_llks_array = np.zeros(n_alleles)
    mh_lpriors_array = np.zeros(n_alleles)
    mh_probabilities_array = np.zeros(n_alleles)
    mh_matrix = np.zeros((n_alleles, n_alleles))
    for a in range(n_alleles):
        genotype_alleles[variable_allele] = a
        mh_options(
            genotype_alleles=genotype_alleles,
            variable_allele=variable_allele,
            haplotypes=haplotypes,
            reads=reads,
            read_counts=read_counts,
            inbreeding=inbreeding,
            llks_array=mh_llks_array,
            lpriors_array=mh_lpriors_array,
            probabilities_array=mh_probabilities_array,
        )
        mh_matrix[a, :] = mh_probabilities_array.copy()
        # the transition probabilities given a single allele
        # should not be the same as the gibbs approach
        assert any(gibbs_probabilities_array != mh_probabilities_array)

    # get long run behavior of MH approach
    mh_matrix = np.linalg.matrix_power(mh_matrix, 1000)
    mh_long_run = mh_matrix[0].copy()

    # assert gibbs and MH longruns are equivalent
    np.testing.assert_array_almost_equal(gibbs_long_run, mh_long_run)
