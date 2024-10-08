import sys
import argparse
import numpy as np
from dataclasses import dataclass

import mchap.io.vcf.infofields as INFO
import mchap.io.vcf.formatfields as FORMAT
import mchap.io.vcf.columns as COLUMN
from mchap.application import call_baseclass
from mchap.application.baseclass import SampleAssemblyError, SAMPLE_ASSEMBLY_ERROR
from mchap.application.arguments import (
    CALL_MCMC_PARSER_ARGUMENTS,
    collect_call_mcmc_program_arguments,
)
from mchap.calling.classes import CallingMCMC
from mchap.calling.exact import genotype_likelihoods
from mchap.jitutils import natural_log_to_log10

from mchap.encoding.integer import minimum_error_correction
from mchap.io import qual_of_prob, vcf


@dataclass
class program(call_baseclass.program):
    mcmc_chains: int = 1
    mcmc_steps: int = 2000
    mcmc_burn: int = 1000
    mcmc_incongruence_threshold: float = 0.60

    @classmethod
    def cli(cls, command):
        """Program initialization from cli command

        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser("MCMC haplotype calling")
        for arg in CALL_MCMC_PARSER_ARGUMENTS:
            arg.add_to(parser)

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])

        # sort argument details
        arguments = collect_call_mcmc_program_arguments(args)
        return cls(cli_command=command, **arguments)

    def call_sample_genotypes(self, data):
        """De novo haplotype assembly of each sample.

        Parameters
        ----------
        data : LocusAssemblyData

        Returns
        -------
        data : LocusAssemblyData
        """
        # get haplotypes and metadata
        haplotypes = data.locus.encode_haplotypes()
        prior_frequencies = data.locus.frequencies
        mask_reference_allele = data.locus.mask_reference_allele
        mask = np.zeros(len(haplotypes), bool)
        mask[0] = mask_reference_allele

        # save allele sequences
        data.columndata[COLUMN.REF] = data.locus.sequence
        data.columndata[COLUMN.ALT] = data.locus.alts
        data.infodata[INFO.REFMASKED] = mask_reference_allele
        data.infodata[INFO.AFPRIOR] = prior_frequencies

        # mask zero frequency haplotypes
        mask |= prior_frequencies == 0

        # remove masked haplotypes from mcmc
        if np.any(mask):
            mcmc_haplotypes = haplotypes[~mask]
            mcmc_prior_frequencies = prior_frequencies[~mask]
            mcmc_haplotype_labels = np.where(~mask)[0]
        else:
            # use all haplotypes
            mcmc_haplotype_labels = None
            mcmc_prior_frequencies = prior_frequencies
            mcmc_haplotypes = haplotypes

        # must have one or more haplotypes for MCMC
        invalid_scenario = len(mcmc_haplotypes) == 0

        # handle invalid scenarios
        # TODO: handle this more elegantly?
        if len(mcmc_haplotypes) == 0:
            # must have one or more haplotypes for MCMC
            invalid_scenario = True
            data.columndata[COLUMN.FILTER].append(vcf.filters.NOA.id)
        elif (prior_frequencies is not None) and np.any(np.isnan(prior_frequencies)):
            # nan caused by zero freq
            invalid_scenario = True
            data.columndata[COLUMN.FILTER].append(vcf.filters.AF0.id)
        else:
            invalid_scenario = False
        if invalid_scenario:
            for sample in data.samples:
                ploidy = data.sample_ploidy[sample]
                data.sampledata[FORMAT.GT][sample] = np.full(ploidy, -1, int)
                data.sampledata[FORMAT.GQ][sample] = np.nan
                data.sampledata[FORMAT.GPM][sample] = np.nan
                data.sampledata[FORMAT.SPM][sample] = np.nan
                data.sampledata[FORMAT.SQ][sample] = np.nan
                data.sampledata[FORMAT.MCI][sample] = np.nan
                data.sampledata[FORMAT.ACP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.AFP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.AOP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.GP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.GL][sample] = np.array([np.nan])
                data.sampledata[FORMAT.MEC][sample] = np.nan
                data.sampledata[FORMAT.MECP][sample] = np.nan
            return data

        # iterate of samples
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                read_calls = data.read_calls[sample]
                read_dists = data.read_dists[sample]
                read_counts = data.read_counts[sample]
                # call haplotypes
                trace = (
                    CallingMCMC(
                        ploidy=data.sample_ploidy[sample],
                        haplotypes=mcmc_haplotypes,
                        inbreeding=data.sample_inbreeding[sample],
                        frequencies=mcmc_prior_frequencies,
                        steps=self.mcmc_steps,
                        chains=self.mcmc_chains,
                        random_seed=self.random_seed,
                    )
                    .fit(
                        reads=read_dists,
                        read_counts=read_counts,
                    )
                    .burn(self.mcmc_burn)
                )
                if mcmc_haplotype_labels is not None:
                    # need to relabel alleles within subset
                    trace = trace.relabel(mcmc_haplotype_labels)
                incongruence = trace.replicate_incongruence(
                    threshold=self.mcmc_incongruence_threshold
                )
                posterior = trace.posterior()
                alleles, genotype_prob, genotype_support_prob = posterior.mode(
                    genotype_support=True
                )

                # store variables
                data.sampledata[FORMAT.GT][sample] = alleles
                data.sampledata[FORMAT.GQ][sample] = qual_of_prob(genotype_prob)
                data.sampledata[FORMAT.GPM][sample] = genotype_prob
                data.sampledata[FORMAT.SPM][sample] = genotype_support_prob
                data.sampledata[FORMAT.SQ][sample] = qual_of_prob(genotype_support_prob)
                data.sampledata[FORMAT.MCI][sample] = incongruence
                mec = np.sum(minimum_error_correction(read_calls, haplotypes[alleles]))
                mec_denom = np.sum(read_calls >= 0)
                mecp = mec / mec_denom if mec_denom > 0 else np.nan
                data.sampledata[FORMAT.MEC][sample] = mec
                data.sampledata[FORMAT.MECP][sample] = mecp

                # posterior allele frequencies/occurrence if requested
                if self.require_AFP():
                    frequencies, counts, occurrence = trace.posterior_frequencies()
                    data.sampledata[FORMAT.ACP][sample] = counts
                    data.sampledata[FORMAT.AFP][sample] = frequencies
                    data.sampledata[FORMAT.AOP][sample] = occurrence

                # genotype posteriors if requested
                if FORMAT.GP in data.formatfields:
                    probabilities = posterior.as_array(len(haplotypes))
                    data.sampledata[FORMAT.GP][sample] = probabilities

                # genotype likelihoods if requested
                if FORMAT.GL in data.formatfields:
                    llks = genotype_likelihoods(
                        reads=read_dists,
                        read_counts=read_counts,
                        ploidy=data.sample_ploidy[sample],
                        haplotypes=haplotypes,
                    )
                    data.sampledata[FORMAT.GL][sample] = natural_log_to_log10(llks)

            # end of try clause for specific sample
            except Exception as e:
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
                raise SampleAssemblyError(message) from e
        return data
