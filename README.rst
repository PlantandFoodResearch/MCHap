MCHap
=====

Polyploid micro-haplotype assembly using Markov chain Monte Carlo simulation.

Installation
------------

MCHap and it's dependencies can be installed from source using pip.
From the root directory of this repository run:

::

    pip install -r requirements.txt
    python setup.py sdist
    pip install dist/mchap-*.tar.gz


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

    mchap assemble ... | bgzip > haplotypes.vcf.gz


A simple example of running ``mchap assemble`` may look like this:

::

    $ mchap assemble \
        --bam file1.bam file2.bam file3.bam \
        --targets loci.bed \
        --variants snps.vcf.gz \
        --reference reference.fasta \
        --ploidy 4 \
        --cores 8 | bgzip > haplotypes.vcf.gz


Note that the input VCF should be indexed with tabix and the bam files
and reference genome should be indexed with samtools.

The `full list of arguments`_ for ``assemble`` can accessed with:

::

    mchap assemble -h


Performance
-----------

The performance of ``mchap assemble`` will depend largely on your data set
but can be tuned using the available parameters.
Generally speaking assembles will be slower for higher ploidy organisms,
with higher read-depths or with more SNPs falling within each locus in the
BED file.

Jit compilation
~~~~~~~~~~~~~~~

MCHap heavily utilizes the numba JIT compiler to speed up MCMC simulations.
However, the first time you run MCHap on a new system it will have to
compile the functions that make use of the numba JIT compiler and the 
compiled functions are then cached for reuse.
This means that MCHap may run a bit slower the first time it's run on a
new system.

Running on multiple threads
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest way to speed up ``mchap assemble`` is to allow it to run on more
that one CPU thread using the ``--cores`` argument.
The maximum number of threads that can be utilized is the number of target
loci specified in the input BED file.
Note that if an assembly is run with more than ``1`` thread then the records
in the output VCF may be in a different order than the loci were in the
input BED file so it may be necessary to sort the VCF file.

Specifying a constant error-rate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, ``mchap assemble`` will use the per base phred-scores found in the
input BAM files to encode read error-rates.
This is a logical default but the variable error-rates limit the ability of 
MCHap to efficiently calculate likelihoods. 
If you know what the expected error-rate for your data is then you can input
this value using the ``--base-error-rate`` and ignore the use of phred scores
with the ``--ignore-base-phred-scores`` flag.
This combination of arguments can significantly improve assembly speed,
especially with higher read depths.

Tuning MCMC parameters
~~~~~~~~~~~~~~~~~~~~~~

The ``mchap assemble`` program uses Markov chain Monte-Carlo (MCMC)
simulations to assemble haplotypes at each locus of each sample.
Reducing the number of steps or complexity of steps will speed up the
assembly bu may lower the reliability of the results.
The number of steps is configured with ``--mcmc-steps`` and the number
that will be removed as burn-in with ``--mcmc-burn``.
It is recommended to remove at least ``100`` steps as burn-in and that
at least ``1000`` steps should be kept to calculate posterior probabilities.

The complexity of steps can also be configured by adjusting the proportion
of structural sub-steps using the ``--mcmc-recombination-step-probability``
and ``--mcmc-partial-dosage-step-probability`` arguments.
These arguments represent the probability that a structural sub-step of
that type will be performed as part of a step in the MCMC simulation.
These sub-steps can be important for convergence so it is not recommended
to reduce their probability any lower than ``0.25``.

There is also an additional parameter called ``--mcmc-dosage-step-probability``
which is used to configure the probability of a "full" dosage-swap sub-step.
This sub-step type is particularly important for identifying the correct
dosage of a genotype and is computationally very simple so its probability
should not be less than ``0.5`` and it is generally recommended to leave it
at its default value of ``1.0``

Fixing SNPs that are likely to be homozygous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As mentioned above, the number of SNPs present in a locus has a significant
impact on the assembly speed.
The ``--mcmc-fix-homozygous`` argument can be used to identify SNPs that
have a high probability of being homozygous and 'fixing' them so that they
do not vary during the assemble process.
This is applied on a per sample bases and so can 'fix' SNPs in one sample
even if they vary in others.
The default value for this argument is ``0.999`` and so it will only 'fix'
SNPs that are extremely unlikely to be heterozygous.
Reducing this value to ``0.99`` could speed up the assembly process but
lowering it too much may result in incorrectly called haplotypes especially
in higher ploidy organisms.



.. _`full list of arguments`: cli-assemble-help.txt
