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
    read_loci, \
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
    mcmc_steps: int = 1000
    mcmc_burn: int = 500
    mcmc_ratio: float = 0.75
    mcmc_alpha: float = 1.0
    mcmc_beta: float = 3.0
    mcmc_fix_homozygous: float = 0.999
    mcmc_allow_recombinations: bool = True
    mcmc_allow_dosage_swaps: bool = True
    depth_filter_threshold: float = 5.0
    probability_filter_threshold: float = 0.95
    kmer_filter_k: int = 3
    kmer_filter_theshold: float = 0.95
    n_cores: int = 1
    precision: int = 3
    chunk_size: int = 10

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
            '--filter-depth',
            type=float,
            nargs=1,
            default=[5.0],
            help=('Minimum sample read depth required to include an assembly result (default = 5.0).')
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
            mcmc_steps=args.mcmc_steps[0],
            mcmc_burn=args.mcmc_burn[0],
            #mcmc_ratio,
            #mcmc_alpha,
            #mcmc_beta,
            #mcmc_fix_homozygous,
            #mcmc_allow_recombinations,
            #mcmc_allow_dosage_swaps,
            depth_filter_threshold=args.filter_depth[0],
            probability_filter_threshold=args.filter_probability[0],
            kmer_filter_k=args.filter_kmer_k[0],
            kmer_filter_theshold=args.filter_kmer[0],
            n_cores=args.cores[0],
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
            vcf.headermeta.phasing('None'),
        )

        filters=(
            vcf.filters.kmer_filter_header(
                k=self.kmer_filter_k,
                threshold=self.kmer_filter_theshold,
            ),
            vcf.filters.depth_filter_header(
                threshold=self.depth_filter_threshold,
            ),
            vcf.filters.prob_filter_header(
                threshold=self.probability_filter_threshold,
            ),
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
            vcf.formatfields.PQ,
            vcf.formatfields.DP,
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

    def _compute_graph(self, header, locus):

        # format data for sample columns in haplotype vcf
        haplotype_vcf_sample_data = {sample: {} for sample in header.samples}
        
        # store called genotype of each phenotype (may have nulls)
        sample_genotype_arrays = {}

        # store genotype distribution within each phenotype
        sample_genotype_dists = {}

        # samples must be in same order as in header
        for sample in header.samples:
            path = self.sample_bam[sample]
            
            # assemble
            read_variants_unsafe = extract_read_variants(locus, path, samples=sample, id='SM')[sample]
            read_variants = add_nan_read_if_empty(locus, read_variants_unsafe[0], read_variants_unsafe[1])
            read_symbols=read_variants[0]
            read_quals=read_variants[1]
            read_calls = encode_read_alleles(locus, read_symbols)
            reads = encode_read_distributions(
                locus, 
                read_calls, 
                read_quals, 
                error_rate=self.read_error_rate, 
                gaps=True,
            )
            trace = DenovoMCMC.parameterize(
                ploidy=self.sample_ploidy[sample], 
                steps=self.mcmc_steps, 
                ratio=self.mcmc_ratio,
                fix_homozygous=self.mcmc_fix_homozygous,
                allow_recombinations=self.mcmc_allow_recombinations,
                allow_dosage_swaps=self.mcmc_allow_dosage_swaps,
            ).fit(reads)

            # posterior dist
            posterior = trace.burn(self.mcmc_burn).posterior()

            # posterior mode phenotype
            mode = posterior.mode_phenotype()
            genotype_dist = mode[0]  # observed genotypes of this phenotype
            genotype_probs = mode[1]  # probs of observed genotypes

            # posterior probability of mode phenotype
            phenotype_probability = np.sum(genotype_probs)
            haplotype_vcf_sample_data[sample]['PPM'] = vcf.formatfields.probabilities(phenotype_probability, self.precision)

            if self.call_best_genotype:
                # call genotype with the highest probability within the mode phenotype
                call_idx = np.argmax(genotype_probs)
                genotype = genotype_dist[call_idx]
                genotype_probability = genotype_probs[call_idx]

            else:
                # most complete genotype of this phenotype given probability threshold (may have nulls)
                call = vcf.call_phenotype(
                    genotype_dist, genotype_probs, 
                    self.probability_filter_threshold
                )
                genotype = call[0]
                genotype_probability = call[1]

            # posterior probability of called genotype
            haplotype_vcf_sample_data[sample]['GPM'] = vcf.formatfields.probabilities(genotype_probability, self.precision)
            
            # depth 
            depth = symbolic.depth(read_symbols)  # also used in filter
            haplotype_vcf_sample_data[sample]['DP'] = vcf.formatfields.haplotype_depth(depth)

            # genotype quality
            haplotype_vcf_sample_data[sample]['GQ'] = vcf.formatfields.quality(genotype_probability) 

            # phenotype quality
            haplotype_vcf_sample_data[sample]['PQ'] = vcf.formatfields.quality(phenotype_probability)

            # filters
            filter_results = vcf.filters.FilterCallSet.new(
                vcf.filters.prob_filter(
                    phenotype_probability,
                    threshold=self.probability_filter_threshold,
                ),
                vcf.filters.depth_haplotype_filter(
                    depth, 
                    threshold=self.depth_filter_threshold,
                ),
                vcf.filters.kmer_haplotype_filter(
                    read_calls, 
                    genotype,
                    k=self.kmer_filter_k,
                    threshold=self.kmer_filter_theshold,
                ),
            )
            haplotype_vcf_sample_data[sample]['FT'] = filter_results

            # store sample genotypes/phenotypes for labeling
            if self.call_filtered:
                # store genotype/phenotype 
                sample_genotype_arrays[sample] = genotype

                # store observed genotypes of mode phenotype
                sample_genotype_dists[sample] = (genotype_dist, genotype_probs)

            else:
                # filter genotype/phenotype and store
                genotype_filtered = vcf.filters.null_filtered_array(genotype, filter_results.failed)
                sample_genotype_arrays[sample] = genotype_filtered

                # store observed genotypes of mode phenotype
                genotype_dist_filtered = vcf.filters.null_filtered_array(genotype_dist, filter_results.failed)
                sample_genotype_dists[sample] = (genotype_dist_filtered, genotype_probs)
        
        # label haplotypes for haplotype vcf
        observed_genotypes = list(sample_genotype_arrays.values())
        labeler = vcf.HaplotypeAlleleLabeler.from_obs(observed_genotypes)
        ref_seq = locus.sequence
        alt_seqs = locus.format_haplotypes(labeler.alt_array())
        allele_counts = labeler.count_obs(observed_genotypes)

        # add genotypes to sample data
        for sample in header.samples:
            genotype = sample_genotype_arrays[sample]
            haplotype_vcf_sample_data[sample]['GT'] = labeler.label(genotype)

            # use labeler to sort gentype probs within phenotype
            dist = sample_genotype_dists[sample]
            genotypes = dist[0]
            probs = dist[1]
            tup = labeler.label_phenotype_posterior(genotypes, probs, expected_dosage=True)
            ordered_probs = tup[1]
            expected_dosage = tup[2]
            haplotype_vcf_sample_data[sample]['MPGP'] = vcf.formatfields.probabilities(ordered_probs, self.precision)
            haplotype_vcf_sample_data[sample]['MPED'] = vcf.formatfields.probabilities(expected_dosage, self.precision)
        
        # construct vcf record
        record = vcf.VCFRecord(
            header,
            chrom=locus.contig, 
            pos=locus.start, 
            id=locus.name, 
            ref=ref_seq, 
            alt=alt_seqs, 
            qual=None, 
            filter=None, 
            info=dict(
                END=locus.stop,
                VP=vcf.vcfstr(np.subtract(locus.positions, locus.start)),
                NS=len(header.samples),
                AC=allele_counts[1:],  # exclude reference count
                AN=np.sum(np.greater(allele_counts, 0)),
            ), 
            format=haplotype_vcf_sample_data,
        )

        return record

    def _worker(self, header, locus, queue):
        line = str(self._compute_graph(header, locus))
        queue.put(line)
        return line

    def _writer(self, queue):
        while True:
            line = queue.get()
            if line == 'KILL':
                break
            sys.stdout.write(line + '\n')
            sys.stdout.flush()

    def run(self):

        header = self.header()
        records = [self._compute_graph(header, locus) for locus in self.loci()]
        return vcf.VCF(header, records)


    def write_lines(self):

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

