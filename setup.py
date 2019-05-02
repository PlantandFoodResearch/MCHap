#! /usr/bin/env python3

import os
from setuptools import setup


def read_file(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


setup(name='haplohelper',
      version='0.0.1',
      author='Tim Millar',
      author_email='tim.millar@plantandfood.co.nz',
      description='Local haplotype manipulation and assembly',
      long_description=read_file('README.rst'),
      packages=['haplohelper'],
      keywords=['biology', 'bioinformatics'],
      classifiers=['Development Status :: 1 - Planning',
                   'Intended Audience :: Science/Research',
                   'Programming Language :: Python :: 3.4',
                   'Topic :: Scientific/Engineering']
      )
