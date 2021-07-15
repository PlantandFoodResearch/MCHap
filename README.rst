MCHap
=====

Polyploid micro-haplotype assembly using Markov chain Monte Carlo simulation.

Installation
------------

MCHap and it's dependencies can be installed from source using pip.
From the root directory of this repository run:

.. code:: bash

    pip install -r requirements.txt
    python setup.py sdist
    pip install dist/mchap-*.tar.gz


Usage
-----

Basic assemble example
~~~~~~~~~~~~~~~~~~~~~~

The wrapper CLI tool is ``mchap`` and assembly sub-tool is ``assemble``.
At minimum this tool requires the following inputs:

- One or more bam files of reads aligned to a reference genome
- A bed file of target genomic loci for assembly
- An indexed VCF file containing a 'reference set' of SNPs
- An indexed fasta file containing the reference genome

The ``assemble`` and ``call`` sub-tools write out an uncompressed VCF file
to standard output.
This should generally be compressed and written to a file:

.. code:: bash

    mchap assemble ... | bgzip > haplotypes.vcf.gz


A simple example of running ``mchap assemble`` may look like this:

.. code:: bash

    $ mchap assemble \
        --bam file1.bam file2.bam file3.bam \
        --targets loci.bed \
        --variants snps.vcf.gz \
        --reference reference.fasta \
        --ploidy 4 \
        --cores 8 | bgzip > haplotypes.vcf.gz


Note that the input VCF should be indexed with tabix and the bam files
and reference genome should be indexed with samtools.

The `full list of arguments`_ for ``assemble`` can be accessed with:

.. code:: bash

    mchap assemble -h


Job array example
~~~~~~~~~~~~~~~~~

The ``assemble`` sub-tool can make use of multiple cores using ``--cores``
argument which uses Pythons build in multiprocessing library.
However this method of parallel processing can be difficult and inefficient
fow large scale assembles in a shared HPC environment.
In such a case it may be more efficient to submit many individual assemblies
each producing a separate output VCF file and then merging the resulting
VCFs.
The following is an example of using the `asub`_ script for array submission
using the ``--region`` and ``--region-id`` parameters to specify each
individual locus as a sub-job:

.. code:: bash

    JOBNAME='myjob'
    VCFDIR="./$JOBNAME.vcf"
    mkdir $VCFDIR
    while read line; do
    contig=$(echo "$line" | cut -f 1)
    start=$(echo "$line" | cut -f 2)
    stop=$(echo "$line" | cut -f 3)
    name=$(echo "$line" | cut -f 4)
    region="$contig:$start-$stop"
    cat << EOF
    mchap assemble \
        --region "$region" \
        --region-id "$name" \
        --variants "$VARIANTS" \
        --reference "$REFERENCE" \
        --sample-bam "$SAMPLE_BAMS" \
        | bgzip > $VCFDIR/$name.vcf.gz
    EOF
    done <"$BEDFILE" | asub -c 100 -j "$JOBNAME"


The above example will iterate though a 4 column bed file (``$BEDFILE``) of
target loci and create a job array with one sub-job per entry in the bed file.
Each sub-job will output a VCF file with a single record in a directory called
"myjob.vcf" (from the ``$VCFDIR`` variable).
Each VCF file will be named based on the locus name in column 4 of the bed file
(hence these names must be unique).
In this example we use the  ``--sample-bam`` option to specify the bam file for
each sample explicitly as this is more efficient when running many small
(sub-) jobs.
Note that the ``--region-id`` argument is only used to set the id of each
variant record in the VCF output and may be omitted in which case the the
variant records will be un-named.


Performance
-----------

The performance of ``mchap assemble`` will depend largely on your data set
but can be tuned using the available parameters.
Generally speaking assembles will be slower for higher ploidy organisms,
with higher read-depths, or with more SNPs falling within each locus in the
BED file.

Jit compilation
~~~~~~~~~~~~~~~

MCHap heavily utilizes the numba JIT compiler to speed up MCMC simulations.
However, the first time you run MCHap on a new system it will have to
compile the functions that make use of the numba JIT compiler and the 
compiled functions are then cached for reuse.
This means that MCHap may run a bit slower the first time it's run after
installation.

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
this value using the ``--base-error-rate`` argument and ignore the use of phred
scores with the ``--ignore-base-phred-scores`` flag.
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
to reduce their probability much lower than ``0.25``.

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
This is applied on a per sample bases and will 'fix' SNPs in one sample
even if they vary in others.
The default value for this argument is ``0.999`` and so it will only 'fix'
SNPs that are extremely unlikely to be heterozygous.
Reducing this value to ``0.99`` could speed up the assembly process but
lowering it too much may result in incorrectly called haplotypes especially
in higher ploidy organisms.

Parallel-tempering
~~~~~~~~~~~~~~~~~~

The ``mchap assemble`` program can use parallel-tempering to reduce the
risk of multi-modality and thereby reduce the chance of incorrectly
assembled haplotypes.
However, parallel-tempering is computationally intensive as an additional
MCMC simulation is run for each additional temperature.
To balance this trade-off it's possible to specify parallel-temperature
on a per-sample basis using the ``--sample-mcmc-temperatures`` parameter.
For example, when assembling haplotypes for samples of a pedigree it may
be desirable to specify multiple temperatures for founding individuals
to ensure that the founding alleles are identified without using
parallel-tempering for all of the progeny derived from those founders.


.. _`full list of arguments`: cli-assemble-help.txt
.. _`asub`: https://github.com/lh3/asub
