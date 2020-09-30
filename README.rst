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


The full list of ``assemble`` arguments can be seen in the help text:

::

    $ mchap assemble -h
    usage: MCMC haplotype assembly [-h] [--targets TARGETS] [--variants VARIANTS]
                                [--reference REFERENCE] [--bam [BAM [BAM ...]]]
                                [--bam-list BAM_LIST] [--ploidy PLOIDY]
                                [--sample-ploidy SAMPLE_PLOIDY]
                                [--sample-list SAMPLE_LIST]
                                [--error-rate ERROR_RATE] [--best-genotype]
                                [--call-filtered] [--mcmc-chains MCMC_CHAINS]
                                [--mcmc-steps MCMC_STEPS]
                                [--mcmc-burn MCMC_BURN]
                                [--mcmc-fix-homozygous MCMC_FIX_HOMOZYGOUS]
                                [--mcmc-seed MCMC_SEED]
                                [--filter-depth FILTER_DEPTH]
                                [--filter-read-count FILTER_READ_COUNT]
                                [--filter-probability FILTER_PROBABILITY]
                                [--filter-kmer-k FILTER_KMER_K]
                                [--filter-kmer FILTER_KMER]
                                [--read-group-field READ_GROUP_FIELD]
                                [--cores CORES]

    optional arguments:
    -h, --help            show this help message and exit
    --targets TARGETS     Bed file containing genomic intervals for haplotype
                            assembly. First three columns (contig, start, stop)
                            are mandatory. If present, the fourth column (id) will
                            be used as the variant id in the output VCF.
    --variants VARIANTS   Tabix indexed VCF file containing SNP variants to be
                            used in assembly. Assembled haplotypes will only
                            contain the reference and alternate alleles specified
                            within this file.
    --reference REFERENCE
                            Indexed fasta file containing the reference genome.
    --bam [BAM [BAM ...]]
                            A list of 0 or more bam files. Haplotypes will be
                            assembled for all samples found within all listed bam
                            files unless the --sample-list parameter is used.
    --bam-list BAM_LIST   A file containing a list of bam file paths (one per
                            line). This can optionally be used in place of or
                            combined with the --bam parameter.
    --ploidy PLOIDY       Default ploidy for all samples (default = 2). This
                            value is used for all samples which are not specified
                            using the --sample-ploidy parameter
    --sample-ploidy SAMPLE_PLOIDY
                            A file containing a list of samples with a ploidy
                            value used to indicate where their ploidy differs from
                            the default value. Each line should contain a sample
                            identifier followed by a tab and then an integer
                            ploidy value.
    --sample-list SAMPLE_LIST
                            Optionally specify a file containing a list of samples
                            to haplotype (one sample id per line). This file also
                            specifies the sample order in the output. If not
                            specified, all samples in the input bam files will be
                            haplotyped.
    --error-rate ERROR_RATE
                            Expected base-call error rate of sequences in addition
                            to base phred scores (default = 0.0). By default only
                            the phred score of each base call is used to calculate
                            its probability of an incorrect call. The --error-rate
                            value is added to that probability.
    --best-genotype       Flag: allways call the best supported complete
                            genotype within a called phenotype. This may result in
                            calling genotypes with a posterior probability less
                            than --filter-probability however a phenotype
                            probability of >= --filter-probability is still
                            required.
    --call-filtered       Flag: include genotype calls for filtered samples.
                            Sample filter tags will still indicate samples that
                            have been filtered. WARNING: this can result in a
                            large VCF file with un-informative genotype calls.
    --mcmc-chains MCMC_CHAINS
                            Number of independent MCMC chains per assembly
                            (default = 2).
    --mcmc-steps MCMC_STEPS
                            Number of steps to simulate in each MCMC chain
                            (default = 1000).
    --mcmc-burn MCMC_BURN
                            Number of initial steps to discard from each MCMC
                            chain (default = 500).
    --mcmc-fix-homozygous MCMC_FIX_HOMOZYGOUS
                            Fix alleles that are homozygous with a probability
                            greater than or equal to the specified value (default
                            = 0.999). The probability of that a variant is
                            homozygous in a sample is assessed independently for
                            each variant prior to MCMC simulation. If an allele is
                            "fixed" it is not allowed vary within the MCMC thereby
                            reducing computational complexity.
    --mcmc-seed MCMC_SEED
                            Random seed for MCMC (default = 42).
    --filter-depth FILTER_DEPTH
                            Minimum sample read depth required to include an
                            assembly result (default = 5.0). Read depth is
                            measured as the mean of read depth across each
                            variable position.
    --filter-read-count FILTER_READ_COUNT
                            Minimum number of read (pairs) required within a
                            target interval in order to include an assembly result
                            (default = 5).
    --filter-probability FILTER_PROBABILITY
                            Minimum sample assembly posterior probability required
                            to call a phenotype i.e. a set of unique haplotypes of
                            unknown dosage (default = 0.95). Genotype dosage will
                            be called or partially called if it also exceeds this
                            threshold. See also the --best-genotype flag.
    --filter-kmer-k FILTER_KMER_K
                            Size of variant kmer used to filter assembly results
                            (default = 3).
    --filter-kmer FILTER_KMER
                            Minimum kmer representation required at each position
                            in assembly results (default = 0.90).
    --read-group-field READ_GROUP_FIELD
                            Read group field to use as sample id (default = "SM").
                            The chosen field determines tha sample ids required in
                            other input files e.g. the --sample-list argument.
    --cores CORES         Number of cpu cores to use (default = 2).
