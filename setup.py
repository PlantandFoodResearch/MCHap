#! /usr/bin/env python3

import os
from setuptools import setup

def read_file(file_name):
    path = os.path.join(os.path.dirname(__file__), file_name)
    with open(path) as f:
        lines = f.readlines()
    return '\n'.join(lines)

VERSION = read_file('mchap/version.py').split("'")[1]

setup(
    name='mchap',
    version=VERSION,
    author='Tim Millar',
    author_email='tim.millar@plantandfood.co.nz',
    description='Polyploid micro-haplotype assembly',
    long_description=read_file('README.rst'),
    entry_points={"console_scripts": ["mchap=mchap.application.cli:main"]},
    packages=[
        'mchap',
        'mchap/application',
        'mchap/assemble',
        'mchap/encoding',
        'mchap/encoding/integer',
        'mchap/encoding/character',
        'mchap/io',
        'mchap/io/vcf',
    ],
    python_requires='>3.7.0',
    keywords=['biology', 'bioinformatics', 'genetics', 'genomics'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ]
    )
