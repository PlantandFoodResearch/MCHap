De novo Assembly
================

*(Last updated for MCHap version 0.7.0)*

Basic inputs
------------

In most situations, a workflow using MCHap will start with the assembly tool called ``mchap assemble``.
This tool assembles micro-haplotypes from single nucleotide variants (SNVs) within specified genomic regions.
The optimal size of these regions will vary between data sets but will usually be no more than several hundred base-positions.
The minimal set of inputs required to run ``mchap assemble`` are:

- A BED file of target genomic loci for assembly.
- One or more BAM files containing aligned reads for each sample.
- A VCF file containing a 'reference set' of known SNVs.
- A Fasta file containing the reference genome used to produce the BAM and VCF files.

The BED file must contain at least the first three columns which are respectively the chromosome/contig, start-position and stop-position of each assembly target.
If a fourth column containing identifiers for each target region is present, then these identifiers will be reported in the ID column of the output VCF file.
Any further columns within the BED file will be ignored. The contents of this file should look similar to the following:

.. code:: bash

    chrom1	10000	10100	locus1
    chrom1	10500	10650	locus2
    chrom2	5100	5220	locus3


The input VCF file is used to specify the reference and alternate alleles of SNVs.
Any insertion, deletion or multi-nucleotide variants within this VCF file will be ignored.
Sample specific data within the input VCF will also be ignored and it is not necessary for this VCF file to contain any sample data.
This VCF file should be compressed and indexed using ``htslib``.


The reference Fasta file is necessary for reporting non-variable loci within each micro-haplotype.
This file should be indexed using the ``faidx`` utility of samtools.
An error may be raised if the reference sequence does not match the reference alleles reported in the input VCF.


In addition to the required inputs described above, ``mchap assemble`` can also accept a range of optional input files.
These are simple tab-delimited plaintext files which are used to specify an input parameter per each sample.
These files are often not necessary when a single value is suitable for all samples within the dataset.
For example, the ploidy of all samples can be specified using the ``--ploidy`` argument and it is only necessary to provide a ``--sample-ploidy`` file for a mixed-ploidy dataset.

The built-in help menu for ``mchap assemble`` will automatically be displayed if the program is called without any arguments e.g.:

.. code:: bash

    mchap assemble

will the `full list of arguments`_ which can be provided to ``mchap assemble``.

Simple example
--------------

A simple example of running ``mchap assemble`` on three tetraploid samples should look something like:

.. code:: bash

    $ mchap assemble \
        --bam sample1.bam sample2.bam sample3.bam \
        --targets targets.bed \
        --variants variants.vcf.gz \
        --reference reference.fasta \
        --ploidy 4 \
        | bgzip > haplotypes.vcf.gz

Note that the backslashes ``\`` specify that the shell command continues on the next line.
This helps format the command but can lead to errors if any white-space is present after a backslash (including windows carriage returns).
By default, MCHap commands will write their output to ``stdout`` (i.e., print the results in the terminal).
In the final line of the above command we use a unix pipe ``|`` to redirect the output of ``mchap assemble`` into the ``bgzip`` utility available in ``htslib``.
The compressed output vcf is then written to a file.

Common parameters
-----------------

In this section we give an overview of some of the more common parameters that can be used by ``mchap assemble``.
Each of these parameters are optional and a default value will be used if they aren't specified.
However, the default parameters will not represent a sensible choice for every situation and it is worth considering what a sensible value should be.

Sample parameters
~~~~~~~~~~~~~~~~~

Sample parameters are used to specify information about each sample.
Some of parameters such as ploidy have obvious importance when calling genotypes,
however, other parameters such as expected inbreeding coefficients can have more subtle effects on the results.

- ``--ploidy``: The ploidy of all samples in the analysis (default = ``2``, must be a positive integer).
  If samples of multiple ploidy levels are present then these can be specified in a file using the ``--sample-ploidy`` argument.
  The ploidy determines the number of alleles called for each sample within the output VCF.
- ``--inbreeding``: The expected inbreeding coefficient of each sample (default = ``0``, must be less than ``1`` and greater than or equal to ``0``).
  If samples have variable inbreeding coefficients then these can be specified in a file using the ``--sample-inbreeding`` argument.
  The inbreeding coefficient is used in combination with allelic variability in the input VCF to determine a prior distribution of genotypes.
  Generally speaking, the higher the expected inbreeding coefficient, the higher the homozygosity of the sample.
  The effect of the inbreeding coefficient (and the prior distribution) is more pronounced with lower read depths.
  It is worth noting that the inbreeding coefficient in rarely ``0`` in real samples, particularly autopolyploids.
  If the genotypes called by MCHap are excessively heterozygous then it is worth considering trying to estimate sample inbreeding coefficients.
  With ``mchap assemble`` in particular, it usually better to slightly over-estimate the inbreeding coefficient rather than underestimating it.
  This is because the ``mchap assemble`` program assumes that samples are derived from a population in which all possible micro-haplotypes are present which can result in higher heterozygosity. 

Output parameters
~~~~~~~~~~~~~~~~~

Output parameters are used to determine what data is reported by MCHap.
These parameters have no effect on the assembly process itself but may be important for downstream analysis.

- ``--report``: Specify one or more optional fields to report in the output VCF file. 
  The available options include:

  * ``AFP``: Posterior mean allele frequencies (One value per unique allele for each sample).
  * ``GP``: Genotype posterior probabilities (One value per possible genotype per sample).
  * ``GL``: Genotype Likelihoods (One value per possible genotype per sample).

  Note that reporting the ``GP`` or ``GL`` fields can result in exceptionally large VCF files.

- ``--haplotype-posterior-threshold``: A threshold value used to determine which micro-haplotypes are reported in the output VCF (default = 0.2).
  This value is compared to the the posterior probability of a given micro-haplotype *occurring* in each sample (irrespective of copy number).
  If the probability of occurrence is greater than, or equal to, the specified threshold in one or more samples then the corresponding micro-haplotype will be reported in the output VCF.
  Hence, a higher threshold value will result in fewer unique haplotypes being reported in the output VCF.
  The exclusion of micro-haplotypes by this threshold value can also result in truncated posterior distributions.
  If a posterior distribution has been truncated then the values of the ``AFP`` and ``GP`` fields may not sum to ``1``.
  Note also that a micro-haplotype may be reported as an alternate allele in the VCF even if it is not called as being present in any of the samples.
  In large populations it can be useful to increase the threshold value to reduce the number of spurious alleles which are reported.
  This is most appropriate with populations of related individuals because they are likely to share alleles resulting in a lower chance of excluding a real micro-haplotype.

Read parameters
~~~~~~~~~~~~~~~

The following parameters determine how MCHap reads and interprets input data from BAM files.
The default values of these parameters are generally suitable for Illumina short read sequences.

- ``--read-group-field``: Read-group field used as sample identifier (default = ``"SM"``).
- ``--base-error-rate``: Expected base-calling error rate for reads (default = ``0.0024``).
  The default value is taken from `Pfeiffer et al (2018)`_.
- ``--mapping-quality``: The minimum mapping quality required for a read to be used (default = ``20``).
- ``--keep-duplicate-reads``: Use reads marked as duplicates in the assembly (these are skipped by default).
- ``--keep-qcfail-reads``: Use reads marked as qcfail in the assembly (these are skipped by default).
- ``--keep-supplementary-reads``: Use reads marked as supplementary in the assembly (these are skipped by default).



.. _`full list of arguments`: ../cli-assemble-help.txt
.. _`Pfeiffer et al (2018)`: https://www.doi.org/10.1038/s41598-018-29325-6
