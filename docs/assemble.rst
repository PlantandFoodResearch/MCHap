MCHap assemble
==============

De novo assembly of micro-haplotypes.

*(Last updated for MCHap version 0.10.0)*

Background
----------

The ``mchap assemble`` tool is used for de novo assembly of micro-haplotypes in one or 
more individuals.
Haplotypes are assembled from aligned reads in BAM files using known SNVs 
(single nucleotide variants) from a VCF file.
A BED file is also required to specify the assembled loci.
The output of ``mchap assemble`` is a VCF file with assembled micro-haplotype variants
and genotype calls (some genotype calls may be incomplete).

``mchap assemble`` uses a Markov chain Monte-Carlo simulation (based on the 
Metropolis-Hastings algorithm) to propose genotypes composed of micro-haplotypes.
This algorithm approximates the posterior genotype distribution of each individual
given its ploidy, inbreeding coefficient, and observed sequence alignments.
The posterior mode genotype is then reported as the genotype call within the
output VCF file.
Note that the genotype calls of different samples are completely independent of
one another.

Micro-haplotype alleles are reported in the output VCF file if there is reasonable
confidence that they occur within one or more samples (this threshold is configurable).
Note that the exclusion of low confidence micro-haplotypes can result in some
(low quality) genotype calls with "unknown" alleles.

Most MCHap workflows will start by using ``mchap assemble`` for de novo micro-haplotype
assembly.
The output of ``mchap assemble`` can then optionally be used as input for ``mchap call``
to improve genotype call accuracy and avoid incomplete genotype calls.
However, the validity of this "two step" approach depends upon the population structure
of samples and purpose of the analysis.
The two step approach is generally more applicable and effective in populations of
closely related samples.

Basic inputs
------------

The minimal set of inputs required to run ``mchap assemble`` are:

- A BED file of target genomic loci for assembly.
- One or more BAM files containing aligned reads for each sample.
- A VCF file containing a 'reference set' of known SNVs.
- A Fasta file containing the reference genome used to produce the BAM and VCF files.

The **BED** file must contain at least three columns which are respectively the 
chromosome/contig, start-position and stop-position of each assembly target.
An optional fourth column can be used to provided identifiers for each locus.
These identifiers will be reported in the ID column of the output VCF.
Any further columns within the BED file will be ignored. 
The contents of this file  should look similar to the following:

.. code:: bash

    chrom1	10000	10100	locus1
    chrom1	10500	10650	locus2
    chrom2	5100	5220	locus3

The optimal size of BED loci will vary based upon allelic variation and experimental
hypothesis, but they should usually be no more than several hundred base-positions.
Larger loci contacting more SNVs will result in greater allelic diversity and
segregation.
Smaller loci with fewer SNVs will result in more robust haplotypes and genotype
calls.

The input **VCF** file is used to specify the reference and alternate alleles of SNVs.
Any insertion, deletion or multi-nucleotide variants within this VCF file will be 
ignored.
Sample data within this VCF file will be ignored and the sample data may even be
omitted from the VCF file entirely. 
This VCF file should be compressed and indexed using ``htslib`` (e.g., compressed 
with ``bgzip`` and indexed with ``tabix``).


The reference **Fasta** file is necessary for reporting non-variable loci within each 
micro-haplotype.
This file should be indexed using the ``faidx`` utility from samtools.
An error will be raised if the reference sequence does not match the reference alleles 
reported in the input VCF.


In addition to the required inputs described above, ``mchap assemble`` can also accept 
a range of optional parameters and/or input files.
Additional input files are simple tab-delimited plaintext files which are used to 
specify an input parameter per each sample.
These files are often not necessary when a single value is suitable for all samples 
within the dataset.
For example, the **ploidy** of all samples can be specified using the ``--ploidy`` argument.
However, the ``--ploidy`` argument can also be the location plaintext file containing
a list of sample identifiers and their corresponding ploidy levels to accommodate
mixed-ploidy datasets.
Each line of this file contains a single sample identifier and its ploidy separated by a
tab.

The built-in help menu for ``mchap assemble`` will automatically be displayed if 
the program is called without any arguments e.g.:

.. code:: bash

    mchap assemble

will the `full list of arguments`_ which can be provided to ``mchap assemble``.

Simple example
--------------

A simple example of running ``mchap assemble`` on three tetraploid samples should 
look something like:

.. code:: bash

    $ mchap assemble \
        --bam sample1.bam sample2.bam sample3.bam \
        --targets targets.bed \
        --variants variants.vcf.gz \
        --reference reference.fasta \
        --ploidy 4 \
        | bgzip > haplotypes.vcf.gz

Note that the backslashes (``\``) specify that the shell command continues on the 
next line.
This helps with formatting the command, but can lead to errors if any white-space
is present after a backslash (including windows carriage returns).
By default, MCHap commands will write their output to ``stdout`` (i.e., print the 
results in the terminal).
In the final line of the above command we use a unix pipe (``|``) to redirect the 
output of ``mchap assemble`` into the ``bgzip`` utility available in ``htslib``.
The compressed output vcf is then written to a file called ``haplotypes.vcf.gz``.

Analyzing many samples
----------------------

Listing each BAM file as part of the command becomes cumbersome when working with a
large number of samples.
The example above can be adapted to use a plaintext file containing a list of BAM file
locations
For example, using a file called ``bam_files.txt`` with contents:

.. code:: bash

    /full/path/to/sample1.bam
    /full/path/to/sample3.bam
    /full/path/to/sample2.bam

The analysis can then be run using:

.. code:: bash

    $ mchap assemble \
        --bam bam_files.txt \
        --targets targets.bed \
        --variants variants.vcf.gz \
        --reference reference.fasta \
        --ploidy 4 \
        | bgzip > haplotypes.vcf.gz

Keeping track of the BAM file relating to each specific sample can be error prone.
If we want to explicitly make sure that we are analyzing the correct samples
then we can also specify sample identifiers in ``bam_files.txt`` followed by a
tab and then the BAM location:

.. code:: bash

    sample_name1	/full/path/to/sample1.bam
    sample_name2	/full/path/to/sample3.bam
    sample_name3	/full/path/to/sample2.bam

If the specified sample name is not found within the associated BAM file then
an error will be raised.

Common parameters
-----------------

In this section we give an overview of some of the more common parameters that 
can be used by ``mchap assemble``.
Each of these parameters are optional and a default value will be used if they 
aren't specified.
However, the default parameters will not represent a sensible choice for every 
situation and it is worth considering what a sensible value should be.

Sample parameters
~~~~~~~~~~~~~~~~~

Sample parameters are used to specify information about each sample.
Some of parameters (e.g., ploidy) have obvious importance when calling genotypes.
However, other parameters such as expected inbreeding coefficients can have more subtle 
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
  A higher inbreeding coefficient will result in increased homozygosity of genotype
  calls.
  This effect is more pronounced with lower read depths and noisier sequencing data.

  It is worth noting that the inbreeding coefficient is rarely ``0`` in real samples, 
  particularly in autopolyploids.
  This means that, by default, MCHap will be biased towards excessively heterozygous
  genotype calls.
  This bias is more pronounced in inbred samples and with lower sequencing depth.
  If the genotype calls output by MCHap appear to be excessively heterozygous,
  it is worth considering if the inbreeding coefficients have been underestimated.

  With ``mchap assemble`` in particular, it usually better to slightly over-estimate the 
  inbreeding coefficient rather than underestimating it.
  This is because the ``mchap assemble`` program assumes that samples are derived from a 
  population in which all *possible* micro-haplotypes are present.
  This assumption is unrealistic for real populations, but is currently unavoidable. 
  
  If samples have variable inbreeding coefficients then these can be specified within a
  file and the location of that file is then passed to the ``--inbreeding`` argument.
  Each line of this file must contain the identifier of a sample and its inbreeding 
  coefficient separated by a tab.

Sample pooling
~~~~~~~~~~~~~~

MCHap allows you to define 'pools' of samples using the ``--sample-pools`` parameter.
A sample pool will combine the reads of its constituent samples but otherwise is treated
identically to a regular sample. Sample parameters relating to the sample pool including
``--ploidy`` and ``--inbreeding`` must be set using the name of the sample pool rather
than its constituent samples. Uses for sample pools include:

- Combining the replicates into a single sample
- Renaming samples (using a pool per sample)
- Combining a set samples that are expected to contain a known number of haplotypes (e.g.,
  the progeny of a bi-parental cross)

The ``--sample-pools`` parameter can specify either the name of a single 'pool' containing
all of the samples, or a tabular file assigning samples to pools. If a tabular fle is used,
each line must contain the name of a sample followed by the name of the pool that sample is
assigned to. All samples must be specified in this file but they can be assigned to a pool
of the same name (i.e., a pool per sample). Samples may be assigned to more than one pool. 

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

- ``--haplotype-posterior-threshold``: A threshold value used to determine which 
  micro-haplotypes are reported in the output VCF (default = 0.2).
  This value is compared to the the posterior probability of a given micro-haplotype 
  *occurring* in each sample (irrespective of copy number).

  A micro-haplotype will always be reported in the output VCF if its probability of
  occurrence (in one or more samples) is greater than or equal to the specified threshold.
  This includes haplotypes that are not actually present in any genotype calls
  (i.e., posterior modes).
  Therefore, increasing the threshold value can significantly reduce the number of
  "noise" haplotypes that are reported, and the size of the output VCF file.
  However, this can also result in more genotypes with unknown alleles, and bias in
  the reported posterior distributions.

  Any genotype call containing a haplotype which has been excluded by this threshold
  will instead contain an the "unknown" allele symbol (``.``).
  For example, ``0/0/1/.`` is a tetraploid genotype call with a single unknown allele.

  Exclusion of micro-haplotypes by the threshold value will result in truncated 
  posterior distributions.
  If a posterior distribution has been truncated then the values of the ``AFP`` and 
  ``GP`` fields will not sum to ``1`` (although minor truncations may be rounded off).

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

The performance of ``mchap assemble`` will largely depend on your data,
but it can be tuned using some of the available parameters.
Generally speaking, ``mchap assemble`` will be slower for higher ploidy organisms,
higher read-depths, and greater numbers SNVs falling within each locus of the
BED file.

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
The maximum *possible* number of cores usable by ``mchap assemble`` is the number of loci
within the input BED file.
This will often mean that ``mchap assemble`` can utilize all available cores.
Note that the resulting VCF file may require sorting when more than one core is used.

On computational clusters, it is often preferable to achieve parallelism within the shell
for better integration with a job-schedular and spreading computation across multiple nodes.
This can be achieved by running multiple MCHap processes on different subsets of the targeted
loci and then merging the resulting VCF files.
The easiest approach with ``mchap assemble`` is to split the input BED file into
multiple smaller files.
Alternatively, a user can specify a single locus with ``mchap assemble`` by using the ``--region``
parameter (and optionally the ``--region-id`` parameter) instead of using a BAM file with
``--targets``. 
This can be used to create an array of single loci jobs.
For example, creating an array of jobs using the `asub`_ script for LSF: 

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

Tuning MCMC parameters
~~~~~~~~~~~~~~~~~~~~~~

The ``mchap assemble`` program uses Markov chain Monte-Carlo (MCMC)
simulations to assemble haplotypes at each locus of each sample.
Reducing the number of steps or complexity of steps will speed up the
assembly but may lower the reliability of the results.
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
This sub-step type is computationally simple and it is particularly important
for correctly calling genotype dosages.
Therefore, it is rarely worth lowering this value from its default of ``1.0``.

Fixing SNVs that are likely to be homozygous
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The number of SNVs present in a locus has a significant impact on the speed
of each MCMC step.
The ``--mcmc-fix-homozygous`` argument can be used to identify SNVs that
have a high probability of being homozygous and 'fixing' them so that they
do not vary during the assemble process.
This is applied on a per sample bases and will 'fix' SNVs in one sample
even if they vary in others.
The default value for this argument is ``0.999`` and so it will only 'fix'
SNVs that are extremely unlikely to be heterozygous.
Lowering this value may speed up the assemblies but can also potential
to bias genotype calls.
It is not recommended to lower this value bellow ``0.99`` if you intend
to use any posterior distribution summary statistics.
It may be worth lowering this value as far as ``0.9`` if you are
only utilizing genotype calls, and are mindful of the potential bias.

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


.. _`full list of arguments`: ../cli-assemble-help.txt
.. _`Pfeiffer et al (2018)`: https://www.doi.org/10.1038/s41598-018-29325-6
.. _`asub`: https://github.com/lh3/asub
