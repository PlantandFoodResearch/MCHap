import numpy as np
import pathlib

from mchap.io import loci
from mchap.io import bam


def test_extract_sample_ids():

    path = pathlib.Path(__file__).parent.absolute()
    paths = [
        str(path / 'data/simple.sample1.bam'),
        str(path / 'data/simple.sample2.bam'),
    ]

    expect = {
        'SAMPLE1': paths[0],
        'SAMPLE2': paths[1],
    }

    actual = bam.extract_sample_ids(
        paths,
        id='SM',
    )

    assert len(expect) == len(actual)
    for k, v in expect.items():
        assert actual[k] == v


def test_extract_read_variants():

    sample = 'SAMPLE1'
    path = pathlib.Path(__file__).parent.absolute() 
    path = str(path / 'data/simple.sample1.bam')

    variants = (
        loci.SNP('CHR1', 6, 7, '.', alleles=('A', 'C')),
        loci.SNP('CHR1', 15, 16, '.', alleles=('A', 'G')),
        loci.SNP('CHR1', 22, 23, '.', alleles=('A', 'C', 'T')),
    )

    locus = loci.Locus(
        contig='CHR1', 
        start=5, 
        stop=25, 
        name='CHR1_05_25', 
        sequence='A' * 20, 
        variants=variants
    )

    expect_chars = np.array(
        [['A', 'A', '-'],
         ['A', 'A', '-'],
         ['C', 'G', '-'],
         ['A', 'G', '-'],
         ['A', 'A', 'A'],
         ['A', 'A', 'A'],
         ['C', 'G', 'C'],
         ['A', 'G', 'T'],
         ['-', 'A', 'A'],
         ['-', 'A', 'A'],
         ['-', 'G', 'C'],
         ['-', 'G', 'T'],
         ['-', 'A', 'A'],
         ['-', 'A', 'A'],
         ['-', 'G', 'C'],
         ['-', 'G', 'T'],
         ['-', '-', 'A'],
         ['-', '-', 'A'],
         ['-', '-', 'C'],
         ['-', '-', 'T']], 
    dtype='<U1')
    expect_quals = np.zeros(expect_chars.shape, dtype = np.int16)
    expect_quals[expect_chars != '-'] = 50

    actual = bam.extract_read_variants(
        locus, 
        path, 
        samples=sample,
        id='SM',
    )
    assert sample in actual
    np.testing.assert_array_equal(actual[sample][0], expect_chars)
    np.testing.assert_array_equal(actual[sample][1], expect_quals)


def test_encode_read_alleles():
    variants = (
        loci.SNP('CHR1', 6, 7, '.', alleles=('A', 'C')),
        loci.SNP('CHR1', 15, 16, '.', alleles=('A', 'G')),
        loci.SNP('CHR1', 22, 23, '.', alleles=('A', 'C', 'T')),
    )

    locus = loci.Locus(
        contig='CHR1', 
        start=5, 
        stop=25, 
        name='CHR1_05_25', 
        sequence='A' * 20, 
        variants=variants
    )

    chars = np.array(
        [['A', 'A', '-'],
         ['A', 'A', '-'],
         ['C', 'G', '-'],
         ['A', 'G', '-'],
         ['A', 'A', 'A'],
         ['A', 'A', 'A'],
         ['C', 'G', 'C'],
         ['A', 'G', 'T'],
         ['-', 'A', 'A'],
         ['-', 'A', 'A'],
         ['-', 'G', 'C'],
         ['-', 'G', 'T'],
         ['-', 'A', 'A'],
         ['-', 'A', 'A'],
         ['-', 'G', 'C'],
         ['-', 'G', 'T'],
         ['-', '-', 'A'],
         ['-', '-', 'A'],
         ['-', '-', 'C'],
         ['-', '-', 'T']], 
        dtype='<U1'
    )

    expect = np.array(
        [[ 0,  0, -1],
         [ 0,  0, -1],
         [ 1,  1, -1],
         [ 0,  1, -1],
         [ 0,  0,  0],
         [ 0,  0,  0],
         [ 1,  1,  1],
         [ 0,  1,  2],
         [-1,  0,  0],
         [-1,  0,  0],
         [-1,  1,  1],
         [-1,  1,  2],
         [-1,  0,  0],
         [-1,  0,  0],
         [-1,  1,  1],
         [-1,  1,  2],
         [-1, -1,  0],
         [-1, -1,  0],
         [-1, -1,  1],
         [-1, -1,  2]], 
        dtype=np.int8
    )

    actual = bam.encode_read_alleles(locus, chars)
    np.testing.assert_array_equal(expect, actual)


def test_encode_read_distributions():

    
    variants = (
        loci.SNP('CHR1', 6, 7, '.', alleles=('A', 'C')),
        loci.SNP('CHR1', 15, 16, '.', alleles=('A', 'G')),
        loci.SNP('CHR1', 22, 23, '.', alleles=('A', 'C', 'T')),
    )

    locus = loci.Locus(
        contig='CHR1', 
        start=5, 
        stop=25, 
        name='CHR1_05_25', 
        sequence='A' * 20, 
        variants=variants
    )

    alleles = np.array(
        [[ 0,  0, -1],
         [ 0,  0, -1],
         [ 1,  1, -1],
         [ 0,  1, -1],
         [-1, -1,  2]], 
        dtype=np.int8
    )

    quals = np.zeros(alleles.shape, dtype=np.int16)
    quals[alleles >= 0] = 40

    expect = np.array([
        [[0.9975, 0.0025, 0. ],
         [0.9975, 0.0025, 0. ],
         [np.nan, np.nan, np.nan]],

        [[0.9975, 0.0025, 0. ],
         [0.9975, 0.0025, 0. ],
         [np.nan, np.nan, np.nan]],

        [[0.0025, 0.9975, 0. ],
         [0.0025, 0.9975, 0. ],
         [np.nan, np.nan, np.nan]],

        [[0.9975, 0.0025, 0. ],
         [0.0025, 0.9975, 0. ],
         [np.nan, np.nan, np.nan]],
        [[np.nan, np.nan, 0. ],
         [np.nan, np.nan, 0. ],
         [0.00125, 0.00125, 0.9975]]
    ])

    actual = bam.encode_read_distributions(
        locus,
        alleles,
        quals,
        error_rate=0.0024,
    )
    np.testing.assert_array_almost_equal(expect, actual)
