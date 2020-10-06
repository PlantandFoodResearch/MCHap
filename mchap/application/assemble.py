import sys
import argparse
import numpy as np
from dataclasses import dataclass
import pysam
from itertools import islice
import multiprocessing as mp

from mchap.assemble.mcmc.denovo import DenovoMCMC
from mchap.encoding import character, integer
from mchap.io import \
    read_bed4, \
    extract_sample_ids, \
    extract_read_variants, \
    add_nan_read_if_empty, \
    encode_read_alleles, \
    encode_read_distributions, \
    qual_of_prob, \
    vcf
from mchap.io.biotargetsfile import read_biotargets

import warnings
warnings.simplefilter('error', RuntimeWarning)

@dataclass
class program(object):
    bed: str
    vcf: str
    ref: str
    bams: list
    samples: list
    sample_ploidy: dict
    call_best_genotype: bool = False
    call_filtered: bool = False
    read_group_field: str = 'SM'
    read_error_rate: float = 0.0
    mcmc_chains: int = 2
    mcmc_steps: int = 1000
    mcmc_burn: int = 500
    mcmc_ratio: float = 0.75
    mcmc_alpha: float = 1.0
    mcmc_beta: float = 3.0
    mcmc_fix_homozygous: float = 0.999
    mcmc_allow_recombinations: bool = True
    mcmc_allow_dosage_swaps: bool = True
    mcmc_full_length_dosage_swap: bool = True
    depth_filter_threshold: float = 5.0
    read_count_filter_threshold: int = 5
    probability_filter_threshold: float = 0.95
    kmer_filter_k: int = 3
    kmer_filter_theshold: float = 0.90
    n_cores: int = 1
    precision: int = 3
    random_seed: int = 42
    cli_command: str=None

    @classmethod
    def cli(cls, command):
        """Program initialisation from cli command
        
        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser(
            'MCMC haplotype assembly'
        )

        parser.add_argument(
            '--targets',
            type=str,
            nargs=1,
            default=[None],
            help=('Bed file containing genomic intervals for haplotype assembly. '
            'First three columns (contig, start, stop) are mandatory. '
            'If present, the fourth column (id) will be used as the variant id in '
            'the output VCF.'),
        )

        parser.add_argument(
            '--variants',
            type=str,
            nargs=1,
            default=[None],
            help=('Tabix indexed VCF file containing SNP variants to be used in '
            'assembly. Assembled haplotypes will only contain the reference and '
            'alternate alleles specified within this file.'),
        )

        parser.add_argument(
            '--reference',
            type=str,
            nargs=1,
            default=[None],
            help='Indexed fasta file containing the reference genome.',
        )

        parser.add_argument(
            '--bam',
            type=str,
            nargs='*',
            default=[],
            help=('A list of 0 or more bam files. '
            'Haplotypes will be assembled for all samples found within all '
            'listed bam files unless the --sample-list parameter is used.'),
        )

        parser.add_argument(
            '--bam-list',
            type=str,
            nargs=1,
            default=[None],
            help=('A file containing a list of bam file paths (one per line). '
            'This can optionally be used in place of or combined with the --bam '
            'parameter.'),
        )

        parser.add_argument(
            '--ploidy',
            type=int,
            nargs=1,
            default=[2],
            help=('Default ploidy for all samples (default = 2). '
            'This value is used for all samples which are not specified using '
            'the --sample-ploidy parameter'),
        )

        parser.add_argument(
            '--sample-ploidy',
            type=str,
            nargs=1,
            default=[None],
            help=(
                'A file containing a list of samples with a ploidy value '
                'used to indicate where their ploidy differs from the '
                'default value. Each line should contain a sample identifier '
                'followed by a tab and then an integer ploidy value.'
            ),
        )

        parser.add_argument(
            '--sample-list',
            type=str,
            nargs=1,
            default=[None],
            help=('Optionally specify a file containing a list of samples to '
                'haplotype (one sample id per line). '
                'This file also specifies the sample order in the output. '
                'If not specified, all samples in the input bam files will '
                'be haplotyped.'),
        )

        parser.add_argument(
            '--error-rate',
            nargs=1,
            type=float,
            default=[0.0],
            help=('Expected base-call error rate of sequences '
            'in addition to base phred scores (default = 0.0). '
            'By default only the phred score of each base call is used to '
            'calculate its probability of an incorrect call. '
            'The --error-rate value is added to that probability.')
        )

        parser.set_defaults(call_best_genotype=False)
        parser.add_argument(
            '--best-genotype',
            dest='call_best_genotype',
            action='store_true',
            help=('Flag: allways call the best supported complete genotype '
            'within a called phenotype. This may result in calling genotypes '
            'with a posterior probability less than --filter-probability '
            'however a phenotype probability of >= --filter-probability '
            'is still required.')
        )

        parser.set_defaults(call_filtered=False)
        parser.add_argument(
            '--call-filtered',
            dest='call_filtered',
            action='store_true',
            help=('Flag: include genotype calls for filtered samples. '
            'Sample filter tags will still indicate samples that have '
            'been filtered. '
            'WARNING: this can result in a large VCF file with '
            'un-informative genotype calls.')
        )

        parser.add_argument(
            '--mcmc-chains',
            type=int,
            nargs=1,
            default=[2],
            help='Number of independent MCMC chains per assembly (default = 2).'
        )

        parser.add_argument(
            '--mcmc-steps',
            type=int,
            nargs=1,
            default=[1000],
            help='Number of steps to simulate in each MCMC chain (default = 1000).'
        )

        parser.add_argument(
            '--mcmc-burn',
            type=int,
            nargs=1,
            default=[500],
            help='Number of initial steps to discard from each MCMC chain (default = 500).'
        )

        parser.add_argument(
            '--mcmc-fix-homozygous',
            type=float,
            nargs=1,
            default=[0.999],
            help=('Fix alleles that are homozygous with a probability greater '
            'than or equal to the specified value (default = 0.999). '
            'The probability of that a variant is homozygous in a sample is '
            'assessed independently for each variant prior to MCMC simulation. '
            'If an allele is "fixed" it is not allowed vary within the MCMC thereby '
            'reducing computational complexity.')
        )

        parser.add_argument(
            '--mcmc-seed',
            type=int,
            nargs=1,
            default=[42],
            help=('Random seed for MCMC (default = 42). ')
        )

        parser.add_argument(
            '--filter-depth',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum sample read depth required to include an assembly '
            'result (default = 5.0). '
            'Read depth is measured as the mean of read depth across each '
            'variable position.')
        )

        parser.add_argument(
            '--filter-read-count',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum number of read (pairs) required within a target '
            'interval in order to include an assembly result (default = 5).')
        )

        parser.add_argument(
            '--filter-probability',
            type=float,
            nargs=1,
            default=[0.95],
            help=('Minimum sample assembly posterior probability required to call '
            'a phenotype i.e. a set of unique haplotypes of unknown dosage '
            '(default = 0.95). '
            'Genotype dosage will be called or partially called if it also exceeds '
            'this threshold. '
            'See also the --best-genotype flag.')
        )

        parser.add_argument(
            '--filter-kmer-k',
            type=int,
            nargs=1,
            default=[3],
            help=('Size of variant kmer used to filter assembly results (default = 3).')
        )

        parser.add_argument(
            '--filter-kmer',
            type=float,
            nargs=1,
            default=[0.90],
            help=('Minimum kmer representation required at each position in assembly '
            'results (default = 0.90).')
        )

        parser.add_argument(
            '--read-group-field',
            nargs=1,
            type=str,
            default=['SM'],
            help=('Read group field to use as sample id (default = "SM"). '
            'The chosen field determines tha sample ids required in other '
            'input files e.g. the --sample-list argument.')
        )

        parser.add_argument(
            '--cores',
            type=int,
            nargs=1,
            default=[2],
            help=('Number of cpu cores to use (default = 2).')
        )

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])

        # bam paths
        bams = args.bam
        if args.bam_list[0]:
            with open(args.bam_list[0]) as f:
                bams += [line.strip() for line in f.readlines()]
        if len(bams) != len(set(bams)):
            raise IOError('Duplicate input bams')

        # samples
        if args.sample_list[0]:
            with open(args.sample_list[0]) as f:
                samples = [line.strip() for line in f.readlines()]
        else:
            # read samples from bam headers
            samples = list(extract_sample_ids(bams, id=args.read_group_field[0]).keys())
        if len(samples) != len(set(samples)):
            raise IOError('Duplicate input samples')
        
        # sample ploidy where it differs from default
        sample_ploidy = dict()
        if args.sample_ploidy[0]:
            with open(args.sample_ploidy[0]) as f:
                for line in f.readlines():
                    sample, ploidy = line.strip().split('\t')
                    sample_ploidy[sample] = int(ploidy)

        # default ploidy
        for sample in samples:
            if sample in sample_ploidy:
                pass
            else:
                sample_ploidy[sample] = args.ploidy[0]

        return cls(
            args.targets[0],
            args.variants[0],
            args.reference[0],
            bams,
            samples,
            sample_ploidy,
            call_best_genotype=args.call_best_genotype,
            call_filtered=args.call_filtered,
            read_group_field=args.read_group_field[0],
            read_error_rate=args.error_rate[0],
            mcmc_chains=args.mcmc_chains[0],
            mcmc_steps=args.mcmc_steps[0],
            mcmc_burn=args.mcmc_burn[0],
            #mcmc_ratio,
            #mcmc_alpha,
            #mcmc_beta,
            mcmc_fix_homozygous=args.mcmc_fix_homozygous[0],
            #mcmc_allow_recombinations,
            #mcmc_allow_dosage_swaps,
            #mcmc_full_length_dosage_swap,
            depth_filter_threshold=args.filter_depth[0],
            read_count_filter_threshold=args.filter_read_count[0],
            probability_filter_threshold=args.filter_probability[0],
            kmer_filter_k=args.filter_kmer_k[0],
            kmer_filter_theshold=args.filter_kmer[0],
            n_cores=args.cores[0],
            cli_command=command,
            random_seed=args.mcmc_seed[0]
        )

    def loci(self):
        bed = read_bed4(self.bed)
        for b in bed:
            yield b.set_sequence(self.ref).set_variants(self.vcf)

    def header(self):

        # io
        samples = self.samples

        with pysam.Fastafile(self.ref) as fasta:
            contigs = tuple(
                vcf.ContigHeader(c, l)
                for c, l in zip(fasta.references, fasta.lengths)
            )

        # define vcf template
        meta_fields=(
            vcf.headermeta.fileformat('v4.3'),
            vcf.headermeta.filedate(),
            vcf.headermeta.source(),
            vcf.headermeta.phasing('None'),
            vcf.headermeta.commandline(self.cli_command),
            vcf.headermeta.randomseed(self.random_seed),
        )

        filters=(
            vcf.filters.SamplePassFilter(),
            vcf.filters.SampleKmerFilter(self.kmer_filter_k, self.kmer_filter_theshold),
            vcf.filters.SampleDepthFilter(self.depth_filter_threshold),
            vcf.filters.SampleReadCountFilter(self.read_count_filter_threshold),
            vcf.filters.SamplePhenotypeProbabilityFilter(self.probability_filter_threshold),
        )

        info_fields=(
            vcf.infofields.AN,
            vcf.infofields.AC,
            vcf.infofields.NS,
            vcf.infofields.END,
            vcf.infofields.SNVPOS,
        )

        format_fields=(
            vcf.formatfields.GT,
            vcf.formatfields.GQ,
            vcf.formatfields.PHQ,
            vcf.formatfields.DP,
            vcf.formatfields.RCOUNT,
            vcf.formatfields.MEC,
            vcf.formatfields.FT,
            vcf.formatfields.GPM,
            vcf.formatfields.PPM, 
            vcf.formatfields.MPGP,
            vcf.formatfields.MPED,
        )

        vcf_header = vcf.VCFHeader(
            meta=meta_fields,
            contigs=contigs,
            filters=filters,
            info_fields=info_fields,
            format_fields=format_fields,
            samples=tuple(samples),
        )

        return vcf_header

    def _assemble_locus(self, header, locus):

        # label sample filters
        kmer_filter = header.filters[1]
        depth_filter = header.filters[2]
        count_filter = header.filters[3]
        prob_filter = header.filters[4]

        # format data for sample columns in haplotype vcf
        sample_data = {sample: {} for sample in header.samples}

        for path in self.bams:

            # extract bam data and subset to sample in sample list
            bam_data = extract_read_variants(locus, path, id=self.read_group_field)
            bam_data = {sample: data for sample, data in bam_data.items()
                        if sample in sample_data}

            for sample, (read_chars, read_quals) in bam_data.items():

                # process read data
                read_count = read_chars.shape[0]
                read_depth = character.depth(read_chars)
                read_chars, read_quals = add_nan_read_if_empty(locus, read_chars, read_quals)
                read_calls = encode_read_alleles(locus, read_chars)
                read_dists = encode_read_distributions(
                    locus, 
                    read_calls, 
                    read_quals, 
                    error_rate=self.read_error_rate, 
                )

                # assemble
                trace = DenovoMCMC.parameterize(
                    ploidy=self.sample_ploidy[sample],
                    steps=self.mcmc_steps,
                    chains=self.mcmc_chains,
                    ratio=self.mcmc_ratio,
                    fix_homozygous=self.mcmc_fix_homozygous,
                    allow_recombinations=self.mcmc_allow_recombinations,
                    allow_dosage_swaps=self.mcmc_allow_dosage_swaps,
                    random_seed=self.random_seed,
                ).fit(read_dists)

                # posterior mode phenotype is a collection of genotypes
                phenotype = trace.burn(self.mcmc_burn).posterior().mode_phenotype()

                # call genotype (array(ploidy, vars), probs)
                if self.call_best_genotype:
                    genotype = vcf.call_best_genotype(*phenotype)
                else:
                    genotype = vcf.call_phenotype(*phenotype, self.probability_filter_threshold)

                # apply filters
                filterset = vcf.filters.FilterCallSet((
                    prob_filter(phenotype[1].sum()),
                    depth_filter(read_depth),
                    count_filter(read_count),
                    kmer_filter(read_calls, genotype[0]),
                ))

                # format fields
                sample_data[sample].update({
                    'GPM': vcf.formatfields.probabilities(genotype[1], self.precision),
                    'PPM': vcf.formatfields.probabilities(phenotype[1].sum(), self.precision),
                    'RCOUNT': read_count,
                    'DP': vcf.formatfields.haplotype_depth(read_depth),
                    'GQ': vcf.formatfields.quality(genotype[1]),
                    'PHQ': vcf.formatfields.quality(phenotype[1].sum()),
                    'MEC': integer.minimum_error_correction(read_calls, genotype[0]).sum(),
                    'FT': filterset
                })

                # Null out the genotype and phenotype arrays
                if (not self.call_filtered) and filterset.failed:
                    genotype[0][:] = -1
                    phenotype[0][:] = -1

                # store sample genotypes/phenotypes for labeling
                sample_data[sample]['genotype'] = genotype
                sample_data[sample]['phenotype'] = phenotype
            
        # label haplotypes for haplotype vcf
        observed_genotypes = [sample_data[sample]['genotype'][0] for sample in header.samples]
        labeler = vcf.HaplotypeAlleleLabeler.from_obs(observed_genotypes)
        ref_seq = locus.sequence
        alt_seqs = locus.format_haplotypes(labeler.alt_array())
        allele_counts = labeler.count_obs(observed_genotypes)

        # count called samples
        n_called_samples = len(header.samples)
        if not self.call_filtered:
            for sample in header.samples:
                if sample_data[sample]['FT'].failed:
                    n_called_samples -= 1

        # data for info column of VCF
        info_data = {
            'END': locus.stop,
            'SNVPOS': vcf.vcfstr(np.subtract(locus.positions, locus.start) + 1),
            'NS': n_called_samples,
            'AC': allele_counts[1:],  # exclude reference count
            'AN': np.sum(np.greater(allele_counts, 0)),
        }

        # add genotypes to sample data
        for sample in header.samples:
            genotype = sample_data[sample]['genotype']
            sample_data[sample]['GT'] = labeler.label(genotype[0])

            # use labeler to sort gentype probs within phenotype
            phenotype = sample_data[sample]['phenotype']
            _, probs, dosage = labeler.label_phenotype_posterior(*phenotype, expected_dosage=True)
            sample_data[sample]['MPGP'] = vcf.formatfields.probabilities(probs, self.precision)
            sample_data[sample]['MPED'] = vcf.formatfields.probabilities(dosage, self.precision)
        
        # construct vcf record
        return vcf.VCFRecord(
            header,
            chrom=locus.contig, 
            pos=locus.start, 
            id=locus.name, 
            ref=ref_seq, 
            alt=alt_seqs, 
            qual=None, 
            filter=None, 
            info=info_data, 
            format=sample_data,
        )

    def run(self):
        header = self.header()
        pool = mp.Pool(self.n_cores)
        jobs = ((header, locus) for locus in self.loci())
        records = pool.starmap(self._assemble_locus, jobs)
        return vcf.VCF(header, records)

    def _worker(self, header, locus, queue):
        line = str(self._assemble_locus(header, locus))
        queue.put(line)
        return line

    def _writer(self, queue):
        while True:
            line = queue.get()
            if line == 'KILL':
                break
            sys.stdout.write(line + '\n')
            sys.stdout.flush()

    def run_stdout(self):

        header = self.header()

        for line in header.lines():
            sys.stdout.write(line + '\n')
        sys.stdout.flush()

        manager = mp.Manager()
        queue = manager.Queue()
        pool = mp.Pool(self.n_cores)

        # start writer process
        writer = pool.apply_async(self._writer, (queue,))

        jobs = []
        for locus in self.loci():
            job = pool.apply_async(self._worker, (header, locus, queue))
            jobs.append(job)

        for job in jobs:
            job.get()

        queue.put('KILL')
        pool.close()
        pool.join()
