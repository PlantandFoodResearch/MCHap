MCHap
=====

MCHap is suite of command line tools for micro-haplotype assembly and genotype
calling in autopolyploids.
The primary components of MCHap are ``mchap assemble`` and ``mchap call``.

Installation
------------

MCHap and it's dependencies can be installed from source using pip.
From the root directory of this repository run:

.. code:: bash

    pip install -r requirements.txt
    python setup.py sdist
    pip install dist/mchap-*.tar.gz

You should then be able to use the command line tool ``mchap`` which is a wrapper
around ``mchap assemble`` and ``mchap call``.

MCHap includes a suite of unit tests which can be run from the root directory of
this repository with:

.. code:: bash

    pytest -v ./


MCHap assemble
--------------

``mchap assemble`` is used for de novo assembly of micro-haplotypes in one or 
more individuals.
Haplotypes are assembled from aligned reads in BAM files using known SNVs 
(single nucleotide variants) from a VCF file.
A BED file is also required to specify the assembled loci.
The output of ``mchap assemble`` is a VCF file with assembled micro-haplotype variants
and genotype calls (some genotype calls may be incomplete).
See the `MCHap assemble documentation`_ for further information..

MCHap call
----------

``mchap call`` is used for (re-) calling genotypes using a set of known 
micro-haplotypes.
Genotypes are called using aligned reads in BAM files and known micro-haplotype alleles
from a VCF file.
The output of ``mchap call`` is a VCF file with micro-haplotype variants
and genotype calls (all genotype calls will be complete).
It is often beneficial to re-call genotypes with ``mchap call`` using the micro-haplotypes
reported by ``mchap assemble``, particularly in populations of related samples.
See the `MCHap call documentation`_ for further information.

MCHap find-snvs
---------------

``mchap find-snvs`` is a simple tool for identifying putative SNVs to use as the basis for
haplotype assembly.
Putative SNVs are identified based on minimum thresholds for allele depths and/or frequencies
estimated from depths.
The output is reported as a simple VCF file which includes allele depths and population allele
frequencies (estimated from the mean of individual frequencies), but no genotype calls.

Example notebook
----------------

See the `example notebook`_ demonstrating genotype calling with MCHap in a bi-parental population.

Experimental features
---------------------

    \:warning: **WARNING: The following tools are highly experimental!!!** :warning:

- ``mchap call-pedigree``: for pedigree informed genotype calling.
- ``mchap atomize``: for converting micro-haplotype calls to phased sets of SNVs.

See the `experimental notebook`_ demonstrating the `call-pedigree` tool as presented at the 2024 `Tools for Polyploids`_ workshop.

Funding
-------

The development of MCHap was partially funded by the "Tools for Polyploids" Specialty Crop Research Initiative
(NIFA USDA SCRI Award # 2020-51181-32156).

.. image:: docs/img/tools-for-polyploids.png
   :width: 300

.. _`MCHap assemble documentation`: docs/assemble.rst
.. _`MCHap call documentation`: docs/call.rst
.. _`example notebook`: docs/example/bi-parental.ipynb
.. _`experimental notebook`: docs/example/bi-parental-pedigree.ipynb
.. _`Tools for Polyploids`: https://www.polyploids.org/
