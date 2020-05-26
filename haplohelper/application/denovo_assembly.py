import argparse
import dask
from dask import delayed
import numpy as np
from dataclasses import dataclass


from haplohelper.io import \
    LociFile, \
    extract_sample_ids, \
    extract_read_variants, \
    encode_read_alleles, \
    encode_read_distributions, \
    qual_of_prob, \
    format_haplotypes

from haplohelper.assemble.mcmc.denovo import DenovoMCMC

from haplohelper.io import vcf
from haplohelper.encoding import symbolic


@dataclass
class program(object):
    loci: list
    sample_bam: dict
    sample_ploidy: dict
    read_group_field: str = 'SM'
    mcmc_steps: int = 1000
    mcmc_burn: int = 500
    mcmc_ratio: float = 0.75
    mcmc_alpha: float = 1.0
    mcmc_beta: float = 3.0
    mcmc_fix_homozygous: float = 0.999
    mcmc_allow_recombinations: bool = True
    mcmc_allow_dosage_swaps: bool = True
    mcmc_allow_deletions: bool = True
    depth_filter_threshold: float = 5.0
    probability_filter_threshold: float = 0.95
    kmer_filter_k: int = 3
    kmer_filter_theshold: float = 0.95
    n_cores: int = 1

    @classmethod
    def cli(cls, command):
        """Program initialisation from cli command
        
        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser(
            'De novo haplotype assembly'
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
            '--loci',
            type=str,
            nargs=1,
            default=[None],
            help='Loci file containing intervals and SNPs.',
        )

        parser.add_argument(
            '--read-group-field',
            nargs=1,
            type=str,
            default=['SM'],
            help='Read group field to use as sample id (default = "SM").'
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

        loci = list(LociFile(args.loci[0]).fetch())

        read_group_field = args.read_group_field[0]
        bams = args.bam
        sample_bam = extract_sample_ids(bams, id=read_group_field)

        ploidy = args.ploidy[0]
        samples = list(sample_bam.keys())
        sample_ploidy = {sample: ploidy for sample in samples}

        return cls(
            loci,
            sample_bam,
            sample_ploidy,
            read_group_field=read_group_field,
            mcmc_steps=args.mcmc_steps[0],
            mcmc_burn=args.mcmc_burn[0],
            #mcmc_ratio,
            #mcmc_alpha,
            #mcmc_beta,
            #mcmc_fix_homozygous,
            #mcmc_allow_recombinations,
            #mcmc_allow_dosage_swaps,
            #mcmc_allow_deletions,
            depth_filter_threshold=args.filter_depth[0],
            probability_filter_threshold=args.filter_probability[0],
            kmer_filter_k=args.filter_kmer_k[0],
            kmer_filter_theshold=args.filter_kmer[0],
            n_cores=args.cores[0],
        )
    

    def run(self):

        # io
        samples = list(self.sample_bam.keys())

        # define vcf template
        vcf_header = vcf.VCFHeader(
            meta=(
                vcf.headermeta.fileformat('v4.3'),
                vcf.headermeta.filedate(),
                vcf.headermeta.phasing('None'),
            ),
            contigs=vcf.contig_headers(self.loci),
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
            ),
            info_fields=(
                vcf.infofields.AN,
                vcf.infofields.AC,
                vcf.infofields.NS,
                vcf.infofields.END,
                vcf.infofields.VP,
            ),
            format_fields=(
                vcf.formatfields.GT,
                vcf.formatfields.GQ,
                vcf.formatfields.DP,
                vcf.formatfields.FT,
            ),
            samples=tuple(samples),
        )

        vcf_template = vcf.VCF(vcf_header, [])

        # assembly
        for locus in self.loci:

            # format data for sample columns in haplotype vcf
            haplotype_vcf_sample_data = {sample: {} for sample in samples}
            
            # need genotypes arrays to lable haplotype alts
            sample_genotype_arrays = {}

            # samples must be in same order as in header
            for sample in samples:
                path = self.sample_bam[sample]
                
                # assemble
                read_variants = delayed(extract_read_variants)(locus, path, samples=sample, id='SM')[sample]
                read_symbols=read_variants[0]
                read_quals=read_variants[1]
                read_calls = delayed(encode_read_alleles)(locus, read_symbols)
                reads = delayed(encode_read_distributions)(locus, read_calls, read_quals, gaps=True)
                trace = delayed(DenovoMCMC.parameterize)(
                    ploidy=self.sample_ploidy[sample], 
                    steps=self.mcmc_steps, 
                    ratio=self.mcmc_ratio,
                    fix_homozygous=self.mcmc_fix_homozygous,
                    allow_recombinations=self.mcmc_allow_recombinations,
                    allow_dosage_swaps=self.mcmc_allow_dosage_swaps,
                    allow_deletions=self.mcmc_allow_deletions,
                ).fit(reads)

                # posterior max (call)
                best = trace.burn(self.mcmc_burn).posterior().mode()
                genotype = best[0]
                probability = best[1]
                sample_genotype_arrays[sample] = genotype
                
                # depth 
                depth = delayed(symbolic.depth)(read_symbols)
                haplotype_vcf_sample_data[sample]['DP'] = delayed(str)(delayed(int)(delayed(np.mean)(depth)))

                # genotype quality
                haplotype_vcf_sample_data[sample]['GQ'] = delayed(str)(delayed(qual_of_prob)(probability))

                # filters
                haplotype_vcf_sample_data[sample]['FT'] = delayed(vcf.filters.FilterCallSet.new)(
                    delayed(vcf.filters.prob_filter)(
                        probability,
                        threshold=self.probability_filter_threshold,
                    ),
                    delayed(vcf.filters.depth_haplotype_filter)(
                        depth, 
                        threshold=self.depth_filter_threshold,
                    ),
                    delayed(vcf.filters.kmer_haplotype_filter)(
                        read_calls, 
                        genotype,
                        k=self.kmer_filter_k,
                        threshold=self.kmer_filter_theshold,
                    ),
                )
            
            # label haplotypes for haplotype vcf
            observed_genotypes = list(sample_genotype_arrays.values())
            labeler = delayed(vcf.HaplotypeAlleleLabeler.from_obs)(observed_genotypes)
            ref_seq = delayed(format_haplotypes)(locus, labeler.ref_array())
            alt_seqs = delayed(format_haplotypes)(locus, labeler.alt_array())
            allele_counts = labeler.count_obs(observed_genotypes)

            # add genotypes to sample data
            for sample in samples:
                genotype = sample_genotype_arrays[sample]
                haplotype_vcf_sample_data[sample]['GT'] = labeler.label(genotype)
            
            # append to haplotype vcf
            vcf_template.append(
                chrom=locus.contig, 
                pos=locus.start, 
                id=locus.name, 
                ref=ref_seq, 
                alt=alt_seqs, 
                qual=None, 
                filter=None, 
                info=dict(
                    END=locus.stop,
                    VP=','.join(str(pos - locus.start) for pos in locus.positions),
                    NS=len(samples),
                    AC=allele_counts[1:],  # exclude reference count
                    AN=delayed(np.sum)(delayed(np.greater)(allele_counts, 0)),
                ), 
                format=haplotype_vcf_sample_data,
            )

        if self.n_cores == 1:
            compute_kwargs =  dict(scheduler="threads")
        else:
            compute_kwargs =  dict(scheduler="processes", num_workers=self.n_cores)

        vcf_result = dask.compute(vcf_template, **compute_kwargs)[0] 
        return vcf_result
