import sys
import argparse
import numpy as np
from dataclasses import dataclass
import pysam
from itertools import islice
import multiprocessing as mp

from haplokit.assemble.mcmc.denovo import DenovoMCMC
from haplokit.encoding import symbolic
from haplokit.io import \
    read_bed4, \
    extract_sample_ids, \
    extract_read_variants, \
    add_nan_read_if_empty, \
    encode_read_alleles, \
    encode_read_distributions, \
    qual_of_prob, \
    vcf, \
    PFEIFFER_ERROR
from haplokit.io.biotargetsfile import read_biotargets

import warnings
warnings.simplefilter('error', RuntimeWarning)

@dataclass
class program(object):
    bed: str
    vcf: str
    ref: str
    sample_bam: dict
    sample_ploidy: dict
    region: str = None
    call_best_genotype: bool = False
    call_filtered: bool = False
    read_group_field: str = 'SM'
    read_error_rate: float = PFEIFFER_ERROR
    mcmc_chains: int = 2
    mcmc_steps: int = 1000
    mcmc_burn: int = 500
    mcmc_ratio: float = 0.75
    mcmc_alpha: float = 1.0
    mcmc_beta: float = 3.0
    mcmc_fix_homozygous: float = 0.999
    mcmc_allow_recombinations: bool = True
    mcmc_allow_dosage_swaps: bool = True
    depth_filter_threshold: float = 5.0
    read_count_filter_threshold: int = 5
    probability_filter_threshold: float = 0.95
    kmer_filter_k: int = 3
    kmer_filter_theshold: float = 0.95
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
            'De novo haplotype assembly'
        )
        parser.add_argument(
            '--btf',
            type=str,
            nargs=1,
            default=[],
            help='Biotargets file with sample ids, bam paths and ploidy.',
        )

        parser.add_argument(
            '--bam',
            type=str,
            nargs='*',
            default=[],
            help='A list of 0 or more bam files.',
        )

        parser.add_argument(
            '--ploidy',
            type=int,
            nargs=1,
            default=[None],
            help='ploidy of all samples',
        )

        parser.add_argument(
            '--bed',
            type=str,
            nargs=1,
            default=[None],
            help='Tabix indexed 4 column Bed file containing (named) genomic intervals.',
        )

        parser.add_argument(
            '--vcf',
            type=str,
            nargs=1,
            default=[None],
            help='Tabix indexed VCF file containing SNP variants.',
        )

        parser.add_argument(
            '--ref',
            type=str,
            nargs=1,
            default=[None],
            help='Indexed fasta file containing reference genome.',
        )

        parser.add_argument(
            '--region',
            type=str,
            nargs=1,
            default=[None],
            help=('Specify a contig region for haplotype assembly '
            'e.g. "contig:start-stop" (Default = None).'),
        )

        parser.set_defaults(call_best_genotype=False)
        parser.add_argument(
            '--best-genotype',
            dest='call_best_genotype',
            action='store_true',
            help='Allways call the best supported compleate genotype within the called phenotype.'
        )

        parser.set_defaults(call_filtered=False)
        parser.add_argument(
            '--call-filtered',
            dest='call_filtered',
            action='store_true',
            help='Include genotypes of filtered samples.'
        )

        parser.add_argument(
            '--read-group-field',
            nargs=1,
            type=str,
            default=['SM'],
            help='Read group field to use as sample id (default = "SM").'
        )

        parser.add_argument(
            '--error-rate',
            nargs=1,
            type=float,
            default=[PFEIFFER_ERROR],
            help='Expected base-call error rate of sequences (default = {}).'.format(PFEIFFER_ERROR)
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
            help='Number of steps to simulate in MCMC (default = 1000).'
        )

        parser.add_argument(
            '--mcmc-burn',
            type=int,
            nargs=1,
            default=[500],
            help='Number of initial steps to discard from MCMC trace (default = 500).'
        )

        parser.add_argument(
            '--mcmc-fix-homozygous',
            type=float,
            nargs=1,
            default=[0.999],
            help=(
                'Fix alleles that are homozygous with a probability greater '
                'than or equal to the specified value (default = 0.999).'
            )
        )

        parser.add_argument(
            '--mcmc-seed',
            type=int,
            nargs=1,
            default=[42],
            help=('Random seed for MCMC (default = 42).')
        )

        parser.add_argument(
            '--filter-depth',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum sample read depth required to include an assembly result (default = 5.0).')
        )

        parser.add_argument(
            '--filter-read-count',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum number of read (pairs) within interval required to include an assembly result (default = 5).')
        )

        parser.add_argument(
            '--filter-probability',
            type=float,
            nargs=1,
            default=[0.95],
            help=('Minimum sample assembly posterior probability required to result (default = 0.95).')
        )

        parser.add_argument(
            '--filter-kmer-k',
            type=int,
            nargs=1,
            default=[3],
            help=('Size of SNP kmer used to filter assembly results (default = 3).')
        )

        parser.add_argument(
            '--filter-kmer',
            type=float,
            nargs=1,
            default=[0.95],
            help=('Minimum kmer representation required at each position in assembly results (default = 0.95).')
        )

        parser.add_argument(
            '--cores',
            type=int,
            nargs=1,
            default=[1],
            help=('Number of cpu cores to use (default = 1).')
        )

        args = parser.parse_args(command[1:])

        # sample bam paths and ploidy
        if args.btf:
            # read from bio-targets file
            btf = read_biotargets(args.btf[0])
            id_column = btf.header.column_names()[0]
            sample_bam = {}
            sample_ploidy = {}
            for d in btf.iter_dicts():
                sample = d[id_column]
                sample_bam[sample] = d['bam']
                sample_ploidy[sample] = d['ploidy']
        else:
            # check bams for samples
            bams = args.bam
            sample_bam = extract_sample_ids(
                bams, 
                id=args.read_group_field[0]
            )
            # use specified ploidy for all samples
            ploidy = args.ploidy[0]
            samples = list(sample_bam.keys())
            sample_ploidy = {sample: ploidy for sample in samples}

        return cls(
            args.bed[0],
            args.vcf[0],
            args.ref[0],
            sample_bam,
            sample_ploidy,
            region=args.region[0],
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
        bed = read_bed4(self.bed, region=self.region)
        for b in bed:
            yield b.set_sequence(self.ref).set_variants(self.vcf)

    def header(self):

        # io
        samples = list(self.sample_bam.keys())

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
            vcf.infofields.VP,
        )

        format_fields=(
            vcf.formatfields.GT,
            vcf.formatfields.GQ,
            vcf.formatfields.PHQ,
            vcf.formatfields.DP,
            vcf.formatfields.RC,
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

        # samples must be in same order as in header
        for sample in header.samples:
            path = self.sample_bam[sample]
            
            # read data
            read_chars, read_quals = extract_read_variants(locus, path, samples=sample, id='SM')[sample]
            read_count = read_chars.shape[0]
            read_depth = symbolic.depth(read_chars)
            read_chars, read_quals = add_nan_read_if_empty(locus, read_chars, read_quals)
            read_calls = encode_read_alleles(locus, read_chars)
            read_dists = encode_read_distributions(
                locus, 
                read_calls, 
                read_quals, 
                error_rate=self.read_error_rate, 
                gaps=True,
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
                'RC': read_count,
                'DP': vcf.formatfields.haplotype_depth(read_depth),
                'GQ': vcf.formatfields.quality(genotype[1]),
                'PHQ': vcf.formatfields.quality(phenotype[1].sum()),
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
            'VP': vcf.vcfstr(np.subtract(locus.positions, locus.start) + 1),
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
