MCHap call
==========

Calling genotypes from known haplotypes.

*(Last updated for MCHap version 0.7.1)*

Background
----------

The ``mchap call`` tool uses a set of known haplotypes to call genotypes across
a group of samples.
``mchap call`` is generally faster and more robust than ``mchap assemble`` so
long as the *real* haplotypes are specified in the inputs.
``mchap call`` cannot identify novel haplotypes that aren't specified in the input.
There are a number of situations where ``mchap call`` can be useful:

- When ancestral/population haplotypes are already known with high confidence.
- When we want to improve upon the genotypes called with ``mchap assemble``.

The first of these is quite self explanatory.
It's possible that we have prior knowledge of which haplotypes are most likely to
occur in our samples and we have enough confidence in that hypothesis that we believ
it to be a more robust approach that de novo assembly from raw data (perhaps the 
new samples have low read depth).

The second of the above situations is a little counter-intuitive.
How can re-calling genotypes using ``mchap call`` produce better results than just
using the genotype calls from ``mchap assemble``?
There are three main reasons:

- Ensuring complete genotypes.
- Exploiting shared alleles.
- Improved MCMC mixing and priors.

Unlike ``mchap assemble``, which will sometimes produce genotype calls with unknown
alleles (refer to the documentation of that tool), ``mchap call`` will always
produce complete genotypes.
This can be very useful in downstream analyses by avoiding filtering or imputing
genotypes for incomplete loci.

The ability to exploit shared alleles can dramatically improve genotype calls, 
particularly amongst closely related genotypes.
In ``mchap assemble``, the genotypes of an individual is called using only the
haplotypes identified from the sequences associated with that individual.
If that sample has poor quality sequencing data then there is a relatively high
chance that the correct haplotypes may not be identified resulting in an 
incomplete or erroneous genotype call.
In ``mchap call`` the genotype of a given sample is called against *all* of the
input haplotypes.
If the input haplotypes come from ``mchap assemble`` then this includes the
haplotypes identified across *all* samples.
This means that the genotype of a low quality sample can be called using
haplotypes from higher quality (related) samples.

Finally, there are several attributes of the method underlying ``mchap call`` that
can produce more robust results than ``mchap assemble``.
The most important of these is that ``mchap call`` is sampling from a smaller
distribution of genotypes than ``mchap assemble`` because the parameter space is
constrained to the set of known haplotypes rather than all possible haplotypes.
Futhermore, ``mchap call`` uses a Gibbs-sampler with better mixing properties
(``mchap assemble`` uses a Metropolis-Hastings sampler due to the potentially
enormous number of possible haplotypes).

Basic inputs
------------

The minimal set of inputs required to run ``mchap call`` are:

- One or more BAM files containing aligned reads for each sample.
- A VCF file containing a 'reference set' of known micro-haplotypes.

The input VCF file is used to specify the known reference and alternate micro-haplotype
alleles.
The variants at any given locus within this file must have a fixed length, however the
length of variants may differ between loci.
Sample data within this VCF file will be ignored and the sample data may even be
omitted from the VCF file entirely. 
This VCF file should be compressed and indexed using ``htslib`` (e.g., compressed 
with ``bgzip`` and indexed with ``tabix``).

In addition to the required inputs described above, ``mchap call`` can also accept 
a range of optional parameters and/or input files.
Additional input files are simple tab-delimited plaintext files which are used to 
specify an input parameter per each sample.
These files are often not necessary when a single value is suitable for all samples 
within the dataset.
For example, the ploidy of all samples can be specified using the ``--ploidy`` argument.
However, if there are samples with differing levels of ploidy then ``--ploidy`` 
argument can be a path to a plaintext file containing a list of sample identifiers 
and corresponding ploidy levels.

The built-in help menu for ``mchap call`` will automatically be displayed if 
the program is called without any arguments e.g.:

.. code:: bash

    mchap call

will the `full list of arguments`_ which can be provided to ``mchap call``.

Simple example
--------------

A simple example of running ``mchap call`` on three tetraploid samples should 
look something like:

.. code:: bash

    $ mchap call \
        --bam sample1.bam sample2.bam sample3.bam \
        --haplotypes haplotypes.vcf.gz \
        --ploidy 4 \
        | bgzip > recalled-haplotypes.vcf.gz

Note that the backslashes (``\``) specify that the shell command continues on the 
next line.
This helps with formatting the command, but can lead to errors if any white-space
is present after a backslash (including windows carriage returns).
By default, MCHap commands will write their output to ``stdout`` (i.e., print the 
results in the terminal).
In the final line of the above command we use a unix pipe (``|``) to redirect the 
output of ``mchap call`` into the ``bgzip`` utility available in ``htslib``.
The compressed output vcf is then written to a file.

When analyzing many samples it is possible to specify a plaintext file containing
a list of bam file location as described in the documentation for `mchap assemble`_.

Sample parameters
~~~~~~~~~~~~~~~~~

Sample parameters are used to specify information about each sample.
Some of parameters such as ploidy have obvious importance when calling genotypes,
however, other parameters such as expected inbreeding coefficients can have more subtle 
effects on the results.

- ``--ploidy``: The ploidy of all samples in the analysis (default = ``2``, must be a 
  positive integer).
  The ploidy determines the number of alleles called for each sample within the output VCF.
  If samples of multiple ploidy levels are present, then these can be specified within a 
  file and the location of that file is then passed to the ``--ploidy`` argument.
  Each line of this file must contain the identifier of a sample and its ploidy separated
  by a tab.
- ``--inbreeding``: The expected inbreeding coefficient of each sample (default = ``0``, 
  must be less than ``1`` and greater than or equal to ``0``).
  The inbreeding coefficient is used in combination with allelic variability in the input 
  VCF to determine a prior distribution of genotypes.
  Generally speaking, the higher the expected inbreeding coefficient, the higher the 
  homozygosity of the sample.
  The effect of the inbreeding coefficient (and the prior distribution) is more pronounced 
  with lower read depths.
  It is worth noting that the inbreeding coefficient in rarely ``0`` in real samples, 
  particularly in autopolyploids.
  If the genotypes called by MCHap are excessively heterozygous then it is worth considering 
  estimating sample inbreeding coefficients and re-running the analysis with those estimates.
  If samples have variable inbreeding coefficients then these can be specified within a
  file and the location of that file is then passed to the ``--inbreeding`` argument.
  Each line of this file must contain the identifier of a sample and its inbreeding 
  coefficient separated by a tab.

Output parameters
~~~~~~~~~~~~~~~~~

Output parameters are used to determine which data are reported by MCHap.
These parameters have no effect on the assembly process itself, but may be important for 
downstream analysis.

- ``--report``: Specify one or more optional fields to report in the output VCF file. 
  The available options include:

  * ``AFP``: Posterior mean allele frequencies (One value per unique allele for each sample).
  * ``GP``: Genotype posterior probabilities (One value per possible genotype per sample).
  * ``GL``: Genotype Likelihoods (One value per possible genotype per sample).

  Note that reporting the ``GP`` or ``GL`` fields can result in exceptionally large VCF 
  files!

Read parameters
~~~~~~~~~~~~~~~

The following parameters determine how MCHap reads and interprets input data from 
BAM files.
The default values of these parameters are generally suitable for Illumina short 
read sequences.

- ``--read-group-field``: Read-group field used as sample identifier (default = ``"SM"``).
- ``--base-error-rate``: Expected base-calling error rate for reads (default = ``0.0024``).
  The default value is taken from `Pfeiffer et al (2018)`_.
- ``--mapping-quality``: The minimum mapping quality required for a read to be used (default = ``20``).
- ``--keep-duplicate-reads``: Use reads marked as duplicates in the assembly (these are skipped by default).
- ``--keep-qcfail-reads``: Use reads marked as qcfail in the assembly (these are skipped by default).
- ``--keep-supplementary-reads``: Use reads marked as supplementary in the assembly (these are skipped by default).

Parallelism
-----------

MCHap has built in support for running on multiple cores.
This is achieved using the ``--cores`` parameter which defaults to ``1``.
The maximum *possible* number of cores usable by ``mchap call`` is the number of loci
within the VCF file specified with ``--haplotypes``.
In practice, this will often mean that ``mchap call`` can utilize all available cores.
Note that the resulting VCF file may require sorting when more than one core is used.

On computational clusters, it is often preferable to achieve parallelism within the shell
for better integration with a job-schedular and spreading computation across multiple nodes.
This can be achieved by running multiple MCHap processes on different subsets of the targeted
loci and then merging the resulting VCF files.
The easiest way to achieve this with ``mchap call`` is to split the input VCF file into
multiple smaller files.
Alternatively, if you are running ``mchap call`` on the output of ``mchap assemble`` and
you have already split your ``mchap assemble`` job into multiple parts then you can
simply run ``mchap call`` on each output of ``mchap assemble`` before combining the results.


.. _`full list of arguments`: ../cli-call-help.txt
.. _`mchap assemble`: assemble.rst
.. _`Pfeiffer et al (2018)`: https://www.doi.org/10.1038/s41598-018-29325-6