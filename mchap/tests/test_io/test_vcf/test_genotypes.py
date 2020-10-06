import pytest
import numpy as np

from mchap.io.vcf import genotypes


@pytest.mark.parametrize('alleles,phased,expect', [
    pytest.param((0,1,2,-1), False, '0/1/2/.', id='un-phased'),
    pytest.param((0,1,2,-1), True, '0|1|2|.', id='phased'),
])
def test_Genotype__str(alleles, phased, expect):
    obj = genotypes.Genotype(alleles, phased=phased)
    actual = str(obj)
    assert actual == expect


def test_Genotype__sorted():
    alleles = (0,-1,2,1)
    obj = genotypes.Genotype(alleles).sorted()
    actual = obj.alleles
    expect = (0,1,2,-1)
    assert actual == expect


def test_call_best_genotype():
    array = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
        [[0, 0, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
    ], dtype=np.int8)
    probs = np.array([0.65, 0.2, 0.1])

    expect = (array[0], probs[0])
    actual = genotypes.call_best_genotype(array, probs)
    np.testing.assert_array_equal(expect[0], actual[0])
    assert expect[1] == actual[1]


@pytest.mark.parametrize('threshold,expect', [
    pytest.param(
        0.99,
        (np.array([
            [0, 0, 0],
            [1, 1, 1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]), 0.95),
        id='99',
    ),
    pytest.param(
        0.9,
        (np.array([
            [0, 0, 0],
            [1, 1, 1],
            [-1, -1, -1],
            [-1, -1, -1],
        ]), 0.95),
        id='90',
    ),
    pytest.param(
        0.8,
        (np.array([
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
            [-1, -1, -1],
        ]), 0.85),
        id='85',
    ),
    pytest.param(
        0.6,
        (np.array([
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ]), 0.65),
        id='65',
    ),
])
def test_call_phenotype(threshold, expect):
    array = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
        [[0, 0, 0],
         [1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]],
    ], dtype=np.int8)
    probs = np.array([0.65, 0.2, 0.1])

    actual = genotypes.call_phenotype(array, probs, threshold=threshold)
    np.testing.assert_array_equal(expect[0], actual[0])
    np.testing.assert_almost_equal(expect[1], actual[1])


def test_HaplotypeAlleleLabeler__from_obs__ref():
    array = np.array([
        [[0, 1, 1],
         [0, 1, 1],
         [1, 1, 1],
         [-1, -1, -1]],
        [[0, 0, 0],
         [0, 0, 0],
         [1 ,1 ,1],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
    ])
    # counts = 4:5:2
    expect = ((0, 0, 0), (1, 1, 1), (0, 1, 1))
    actual = genotypes.HaplotypeAlleleLabeler.from_obs(array).alleles
    assert actual == expect


def test_HaplotypeAlleleLabeler__from_obs__no_ref():
    array = np.array([
        [[0, 1, 1],
         [0, 1, 1],
         [1, 1, 1],
         [-1, -1, -1]],
        [[1, 0, 0],
         [1, 0, 0],
         [1 ,1 ,1],
         [1, 1, 1]],
        [[1, 0, 0],
         [1, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
    ])
    # counts = 0:5:4:2
    expect = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))
    actual = genotypes.HaplotypeAlleleLabeler.from_obs(array).alleles
    assert actual == expect


def test_HaplotypeAlleleLabeler__count_obs():
    array = np.array([
        [[0, 1, 1],
         [0, 1, 1],
         [1, 1, 1],
         [-1, -1, -1]],
        [[1, 0, 0],
         [1, 0, 0],
         [1 ,1 ,1],
         [1, 1, 1]],
        [[1, 0, 0],
         [1, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
    ])
    obj = genotypes.HaplotypeAlleleLabeler.from_obs(array)
    actual = obj.count_obs(array)
    expect = np.array([0, 5, 4, 2])
    np.testing.assert_array_equal(actual, expect)


def test_HaplotypeAlleleLabeler__argsort():
    alleles = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))
    obj = genotypes.HaplotypeAlleleLabeler(alleles)
    array = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [-1, -1, -1]
    ])
    actual = obj.argsort(array)
    expect = np.array([2, 0, 1, 3])
    np.testing.assert_array_equal(actual, expect)


def test_HaplotypeAlleleLabeler__label():
    alleles = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))
    obj = genotypes.HaplotypeAlleleLabeler(alleles)
    array = np.array([
        [0, 1, 1],
        [0, 1, 1],
        [1, 1, 1],
        [-1, -1, -1]
    ])
    actual = obj.label(array)
    expect = genotypes.Genotype((1, 3, 3, -1))
    assert actual == expect


def test_HaplotypeAlleleLabeler__label_phenotype_posterior():
    alleles = ((0, 0, 0), (1, 1, 1), (1, 0, 0), (0, 1, 1))
    obj = genotypes.HaplotypeAlleleLabeler(alleles)

    array = np.array([
        [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0],
         [1, 1, 1]],
        [[0, 0, 0],
         [0, 0, 0],
         [1, 1, 1],
         [1, 1, 1]],
    ], dtype=np.int8)
    probs = np.array([0.7, 0.1])

    expect_genotypes = [
        genotypes.Genotype((0, 0, 0, 1)),
        genotypes.Genotype((0, 0, 1, 1)),
        genotypes.Genotype((0, 1, 1, 1)),
    ]
    expect_probs = [0.7, 0.1, 0.0]
    expect_dosage = [2.875, 1.125]

    # all output
    actual = obj.label_phenotype_posterior(
        array, 
        probs, 
        unobserved=True, 
        expected_dosage=True
    )
    assert actual == (expect_genotypes, expect_probs, expect_dosage)

    # minimal output
    actual = obj.label_phenotype_posterior(
        array, 
        probs, 
        unobserved=False, 
        expected_dosage=False
    )
    assert actual == (expect_genotypes[0:2], expect_probs[0:2])
