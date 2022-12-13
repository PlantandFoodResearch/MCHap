MCHap
=====

MCHap is a software package for micro-haplotype assembly in auto-polyploids.
The package itself is composed of multiple sub-tools including ``mchap assemble`` 
and ``mchap call``.

``mchap assemble`` is used for de novo assembly of micro-haplotypes in one or 
more individuals.
Haplotypes are assembled from aligned reads using a user specified collection 
of known SNVs (single nucleotide variants).
For more details refer to the `MCHap assemble documentation`_.

``mchap call`` is used for (re-) calling genotypes using a set of known 
micro-haplotypes.
This also requires aligned reads for each individual.
For more details refer to the `MCHap call documentation`_.

.. _`MCHap assemble documentation`: assemble.rst
.. _`MCHap call documentation`: call.rst
