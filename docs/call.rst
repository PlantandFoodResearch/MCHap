MCHap call
==========

Calling genotypes from known haplotypes.

*(Last updated for MCHap version 0.11.0)*

Background
----------

The ``mchap call`` tool uses a set of known haplotypes to call genotypes in one
or more samples.

``mchap call`` uses a Markov chain Monte-Carlo simulation (based on a 
Gibbs-sampler algorithm) to propose genotypes composed of known micro-haplotypes.
This algorithm approximates the posterior genotype distribution of each individual
given its ploidy and observed sequence alignments.
Unlike ``mchap assemble`` which proposes genotypes from a prior distribution
containing all *possible* haplotypes, ``mchap call`` proposes genotypes from
a prior distribution constrained to a set of *known* haplotypes.
The posterior mode genotype is then reported as the called genotype within the
output VCF file.
``mchap call`` is generally faster and more robust than ``mchap assemble`` so
long as the *real* haplotypes are specified in the inputs.

The genotype calls of different samples are independent of one another with
the exception that they are constrained to the same set of known haplotypes.
Therefore, the independence of genotype calls among samples depends upon the
method used to identify the set of known haplotypes.
There two main situations in which ``mchap call`` can be useful:

- When ancestral/population haplotypes are already known with high confidence.
- When we want to improve upon the genotypes called with ``mchap assemble``.

The first situation is mostly self explanatory.
It's possible that we have prior knowledge of which haplotypes are most likely to
occur in our samples and we have enough confidence in that hypothesis that we believe
it to be a more robust approach that de novo assembly from raw data (perhaps the 
new samples have low read depth).

The second situation is a little less intuitive.
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
In ``mchap call``, the genotype of a given sample is called against *all* of the
input haplotypes.
If the input haplotypes come from ``mchap assemble``, then this constrains the
parameter space (i.e., the prior distribution) to haplotypes observed within the
sample population.
Assuming that the haplotypes of each sample can be identified from that
sample *or* another sample in the population (e.g., due to being related),
then this will constrain the prior distribution to a small set of haplotypes
that are likely to include all the relevant haplotypes.
The more closely related the individuals are, the smaller this set of haplotypes
will be.
This effectively uses the information shared among samples to improve genotype
calling.

Finally, there are several attributes of the Gibbs-sampler algorithm used
by ``mchap call`` that can produce more robust results than ``mchap assemble``.
The first of these, as already outlined, is constraining the prior distribution.
This can be further constrained by setting a prior for population allele 
frequencies (described in the following sections).
The second advantage of the Gibbs-sampler is that it proposes new genotypes
by replacing entire haplotypes at each step.
This improves convergence of the MCMC resulting in a more robust approximation
of the posterior distribution.

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

Common parameters
-----------------

Sample parameters
~~~~~~~~~~~~~~~~~

Sample parameters are used to specify information about each sample.
Some of parameters such as ploidy have obvious importance when calling genotypes,
however, other parameters can have more subtle effects on the results.

- ``--reference``: Specify a reference genome. This is optional but highly recommended when
  working with CRAM files instead of BAM files. Specifying the reference genome speeds up
  reading data from CRAM files. It also may be necessary if the CRAM files link to a missing
  reference genome.

- ``--ploidy``: The ploidy of all samples in the analysis (default = ``2``, must be a 
  positive integer).
  The ploidy determines the number of alleles called for each sample within the output VCF.
  
  If samples of multiple ploidy levels are present, then these can be specified within a 
  file and the location of that file is then passed to the ``--ploidy`` argument.
  Each line of this file must contain the identifier of a sample and its ploidy separated
  by a tab.

Sample pooling
~~~~~~~~~~~~~~

MCHap allows you to define 'pools' of samples using the ``--sample-pools`` parameter.
A sample pool will combine the reads of its constituent samples but otherwise is treated
identically to a regular sample. Sample parameters relating to the sample pool including
``--ploidy`` and inbreeding parameter of ``--use-dirmul-prior`` must be set using the name
of the sample pool rather than its constituent samples. Uses for sample pools include:

- Combining the replicates into a single sample
- Renaming samples (using a pool per sample)
- Combining a set samples that are expected to contain a known number of haplotypes (e.g.,
  the progeny of a bi-parental cross)

The ``--sample-pools`` parameter can specify either the name of a single 'pool' containing
all of the samples, or a tabular file assigning samples to pools. If a tabular fle is used,
each line must contain the name of a sample followed by the name of the pool that sample is
assigned to. All samples must be specified in this file but they can be assigned to a pool
of the same name (i.e., a pool per sample). Samples may be assigned to more than one pool. 

Prior distribution and inbreeding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From version ``0.11.0`` of MCHap, the ``mchap call`` tool defaults to a flat prior over genotypes.
In previous versions of MCHap the default behavior was to use Dirichlet-multinomial prior
which *optionally* incorporated the expected inbreeding coefficient of each genotype and a prior
on allele frequencies. However, it is only reasonable to use a Dirichlet-multinomial prior when
meaningful inbreeding coefficients and prior allele frequencies are specified. A flat prior over
genotypes is generally a better choice if inbreeding coefficients and prior allele frequencies
are unknown.

A Dirichlet-multinomial prior can be specified with ``--use-dirmul-prior`` which expects two values.
The first of these is the inbreeding coefficient and the second is the identifier of an INFO field
in the input VCF file to be used as prior allele frequencies.
The inbreeding coefficient may be a single floating point value to be used for all samples, or the
name of a tabular file containing a inbreeding coefficient for each sample.
Each line of this file must contain the identifier of a sample and its inbreeding 
coefficient separated by a tab.
The allele frequency INFO field must contain a single numerical value for each allele (including the
reference allele) and those values will be normalized to ensure that they sum to 1. For example:

.. code:: bash

    $ mchap call \
        --bam sample1.bam sample2.bam sample3.bam \
        --haplotypes haplotypes.vcf.gz \
        --ploidy 4 \
        --use-dirmul-prior 0.1 AFP \
        | bgzip > recalled-haplotypes.vcf.gz

In the above example we specify the posterior allele frequencies (``AFP``) field that
can be optionally output from ``mchap assemble`` is used as the prior allele frequency
distribution for ``mchap call``.

Output parameters
~~~~~~~~~~~~~~~~~

Output parameters are used to determine which data are reported by MCHap.
These parameters have no effect on the assembly process itself, but may be important for 
downstream analysis. MCHap has a range of optional ``INFO`` and ``FORMAT`` parameters that
can be reported with the ``--report`` argument:

- Optional fields:

  * ``INFO/AFPRIOR``: The prior allele frequencies used for genotype calls.
  * ``INFO/ACP``: Posterior mean allele counts of the population (One value per unique allele).
  * ``INFO/AFP``: Posterior mean allele frequencies of the population (One value per unique allele).
  * ``INFO/AOP``: Posterior probability of allele occurring in the population (One value per unique allele).
  * ``INFO/AOPSUM``: Posterior estimate of the number of samples containing an allele (One value per unique allele).
  * ``INFO/SNVDP``: Total read depth at each SNV position withing the assembled locus (One value per SNV).
  * ``FORMAT/ACP``: Posterior mean allele counts (One value per unique allele for each sample).
  * ``FORMAT/AFP``: Posterior mean allele frequencies (One value per unique allele for each sample).
    The mean posterior allele frequency across all samples will be reported as an INFO field.
  * ``FORMAT/AOP``: Posterior probability of allele occurring in a sample (One value per unique allele for each sample).
    The probability of each allele occurring across all samples will be reported as an INFO field.
  * ``FORMAT/GP``: Genotype posterior probabilities (One value per possible genotype per sample).
  * ``FORMAT/GL``: Genotype Likelihoods (One value per possible genotype per sample).
  * ``FORMAT/SNVDP``: Read depth at each SNV position withing the assembled locus (One value per SNV).

- Examples:

  * ``--report INFO/AFP``: will report the ``INFO/AFP`` field.
  * ``--report AFP``: will report the ``INFO/AFP`` and ``FORMAT/AFP`` fields.
  * ``--report INFO/AOP FORMAT/AFP``: will report the ``INFO/AOP`` and ``FORMAT/AFP`` fields.

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

Performance
-----------

The performance of ``mchap call`` will largely depend on your data,
but it can be tuned using some of the available parameters.
Generally speaking, ``mchap call`` will be slower for higher ploidy organisms,
higher read-depths, and greater numbers of alleles within the input VCF file.

Jit compilation
~~~~~~~~~~~~~~~

MCHap heavily utilizes the numba JIT compiler to speed up MCMC simulations.
Numba will compile many functions when MCHap is run for the first time after installation
and the compiled functions will be cached for reuse. 
This means that MCHap may be noticeably slower the first time that it's run after
installation.

Parallelism
~~~~~~~~~~~

MCHap has built in support for running on multiple cores.
This is achieved using the ``--cores`` parameter which defaults to ``1``.
The maximum *possible* number of cores usable by ``mchap call`` is the number of loci
within the VCF file specified with ``--haplotypes``.
This will often mean that ``mchap call`` can utilize all available cores.
Note that the resulting VCF file may require sorting when more than one core is used.

On computational clusters, it is often preferable to achieve parallelism within the shell
for better integration with a job-schedular and spreading computation across multiple nodes.
This can be achieved by running multiple MCHap processes on different subsets of the targeted
loci and then merging the resulting VCF files.
The easiest approach with ``mchap call`` is to split the input VCF file into
multiple smaller files.
Alternatively, if you are running ``mchap call`` on the output of ``mchap assemble`` and
you have already split your ``mchap assemble`` job into multiple parts, then you can
simply run ``mchap call`` on each output of ``mchap assemble`` before combining the results.

Tuning MCMC parameters
~~~~~~~~~~~~~~~~~~~~~~

The ``mchap call`` program uses Markov chain Monte-Carlo (MCMC)
simulations to assemble haplotypes at each locus of each sample.
Reducing the number of steps will speed up the analysis but may lower
the reliability of the results.
The number of steps is configured with ``--mcmc-steps`` and the number
that will be removed as burn-in with ``--mcmc-burn``.
It is recommended to remove at least ``100`` steps as burn-in and that
at least ``1000`` steps should be kept to calculate posterior probabilities.

Excluding input haplotypes
~~~~~~~~~~~~~~~~~~~~~~~~~~

The speed of each MCMC step in ``mchap call`` is largely dependant on the
ploidy of an individual and the number of unique haplotypes in the input VCF file.
Therefore, the speed of analysis can be improved by minimizing unnecessary
haplotypes from the input VCF file.
Depending on population structure and how that input file was generated,
it can be sensible to remove rare haplotypes that are likely to be erroneous.
This can be achieved with the ``--filter-input-haplotypes`` argument.
This argument expects a string which is used to define a filter for alleles.
This string takes the form ``"<field><operator><value>"`` where ``<field>``
is the name of an INFO field, ``<operator>`` is one of {``=``, ``<``, ``>``,
``<=``, ``>=``, ``!=``}, and ``<value>`` is a numerical value.
The  INFO field must be a numerical field with a length of ``R`` (alleles) or
``A`` (alternate alleles).
If reference allele is filtered, then it is included in the output with the
``REFMASKED`` tag.
If the filter field has length ``A`` (alternate alleles), then the filter is not
applied to the reference allele.

An example of using these parameters to exclude rare haplotypes may look like:

.. code:: bash

    $ mchap call \
        --bam sample1.bam sample2.bam sample3.bam \
        --haplotypes haplotypes.vcf.gz \
        --ploidy 4 \
        --filter-input-haplotypes 'AFP>=0.01' \
        | bgzip > recalled-haplotypes.vcf.gz

which will exclude any haplotypes with a frequency of less than ``0.01``.

.. _`full list of arguments`: ../cli-call-help.txt
.. _`mchap assemble`: assemble.rst
.. _`Pfeiffer et al (2018)`: https://www.doi.org/10.1038/s41598-018-29325-6
