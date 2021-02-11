MCHap
=====

Polyploid micro-haplotype assembly using Markov chain Monte Carlo simulation.

Usage
-----

The wrapper CLI tool is ``mchap`` and assembly sub-tool is ``assemble``.
At minimum this tool requires the following inputs:

- One or more bam files of reads aligned to a reference genome
- A bed file of target genomic loci for assembly
- An indexed VCF file containing a 'reference set' of SNPs
- An indexed fasta file containing the reference genome

The ``assemble`` sub-tool writes out an uncompressed VCF file to standard output.
This should generally be compressed and written to a file:

::

    $ mchap assemble ... | bgzip > haplotypes.vcf.gz


The `full list of arguments`_ for ``assemble`` can accessed with:

::

    $ mchap assemble -h


.. _`full list of arguments`: cli-assemble-help.txt
