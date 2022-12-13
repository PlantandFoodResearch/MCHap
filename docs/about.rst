MCHap
=====

MCHap is a software package for micro-haplotype assembly in auto-polyploids.
The package itself is composed of multiple sub-tools including ``mchap assemble`` 
and ``mchap call``.

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

MCHap included a suite of unit tests  which can be run from the root directory of
the repository with:

.. code:: bash

    pytest -v ./


MCHap assemble
--------------

``mchap assemble`` is used for de novo assembly of micro-haplotypes in one or 
more individuals.
Haplotypes are assembled from aligned reads using a user specified collection 
of known SNVs (single nucleotide variants).
For more details refer to the `MCHap assemble documentation`_.

MCHap call
----------

``mchap call`` is used for (re-) calling genotypes using a set of known 
micro-haplotypes.
This also requires aligned reads for each individual.
For more details refer to the `MCHap call documentation`_.

.. _`MCHap assemble documentation`: assemble.rst
.. _`MCHap call documentation`: call.rst
