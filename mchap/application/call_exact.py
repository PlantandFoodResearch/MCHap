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
    CALL_EXACT_PARSER_ARGUMENTS,
    collect_call_exact_program_arguments,
)
from mchap.calling.exact import (
    posterior_mode,
    genotype_likelihoods,
    genotype_posteriors,
    alternate_dosage_posteriors,
    posterior_allele_frequencies,
)
from mchap.jitutils import (
    natural_log_to_log10,
    index_as_genotype_alleles,
)

from mchap.encoding.integer import minimum_error_correction
from mchap.io import qual_of_prob, vcf


@dataclass
class program(call_baseclass.program):
    @classmethod
    def cli(cls, command):
        """Program initialization from cli command

        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser("Exact haplotype calling")
        for arg in CALL_EXACT_PARSER_ARGUMENTS:
            arg.add_to(parser)

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])

        # sort argument details
        arguments = collect_call_exact_program_arguments(args)
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
        haplotypes = data.locus.encode_haplotypes()
        mask_reference_allele = data.locus.mask_reference_allele
        prior_frequencies = data.locus.frequencies

        # save allele sequences
        data.columndata[COLUMN.REF] = data.locus.sequence
        data.columndata[COLUMN.ALT] = data.locus.alts
        data.infodata[INFO.REFMASKED] = mask_reference_allele
        data.infodata[INFO.AFPRIOR] = prior_frequencies

        # check prior frequencies arguments
        if mask_reference_allele:
            assert (prior_frequencies[0] == 0) or np.isnan(prior_frequencies[0])

        # handle invalid scenarios
        # TODO: handle this more elegantly?
        if mask_reference_allele and len(haplotypes) == 1:
            # only allele is masked
            invalid_scenario = True
            data.columndata[COLUMN.FILTER].append(vcf.filters.NOA.id)
        elif np.any(np.isnan(prior_frequencies)):
            # nan caused by zero freq
            invalid_scenario = True
            data.columndata[COLUMN.FILTER].append(vcf.filters.AF0.id)
        else:
            invalid_scenario = False

        # mock data for invalid scenario
        if invalid_scenario:
            for sample in data.samples:
                ploidy = data.sample_ploidy[sample]
                data.sampledata[FORMAT.GT][sample] = np.full(ploidy, -1, int)
                data.sampledata[FORMAT.GQ][sample] = np.nan
                data.sampledata[FORMAT.GPM][sample] = np.nan
                data.sampledata[FORMAT.PHPM][sample] = np.nan
                data.sampledata[FORMAT.PHQ][sample] = np.nan
                data.sampledata[FORMAT.MCI][sample] = np.nan
                data.sampledata[FORMAT.ACP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.AFP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.AOP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.GP][sample] = np.array([np.nan])
                data.sampledata[FORMAT.GL][sample] = np.array([np.nan])
                data.sampledata[FORMAT.MEC][sample] = np.nan
                data.sampledata[FORMAT.MECP][sample] = np.nan
            return data

        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                ploidy = data.sample_ploidy[sample]
                inbreeding = data.sample_inbreeding[sample]
                read_calls = data.read_calls[sample]
                read_dists = data.read_dists[sample]
                read_counts = data.read_counts[sample]

                # call haplotypes
                if (FORMAT.GL in data.formatfields) or (FORMAT.GP in data.formatfields):

                    # calculate full arrays
                    llks = genotype_likelihoods(
                        reads=read_dists,
                        read_counts=read_counts,
                        haplotypes=haplotypes,
                        ploidy=ploidy,
                    )
                    probabilities = genotype_posteriors(
                        log_likelihoods=llks,
                        ploidy=ploidy,
                        n_alleles=len(haplotypes),
                        inbreeding=inbreeding,
                        frequencies=prior_frequencies,
                    )
                    idx = np.argmax(probabilities)
                    alleles = index_as_genotype_alleles(idx, ploidy)
                    genotype_prob = probabilities[idx]
                    _, phenotype_probs = alternate_dosage_posteriors(
                        alleles, probabilities
                    )
                    phenotype_prob = phenotype_probs.sum()

                    # store specified arrays
                    if self.require_AFP():
                        freqs, counts, occur = posterior_allele_frequencies(
                            probabilities, ploidy, len(haplotypes)
                        )
                        data.sampledata[FORMAT.ACP][sample] = counts
                        data.sampledata[FORMAT.AFP][sample] = freqs
                        data.sampledata[FORMAT.AOP][sample] = occur
                    if FORMAT.GL in data.formatfields:
                        data.sampledata[FORMAT.GL][sample] = natural_log_to_log10(llks)
                    if FORMAT.GP in data.formatfields:
                        data.sampledata[FORMAT.GP][sample] = probabilities

                else:
                    # use low memory calculation
                    mode_results = posterior_mode(
                        reads=read_dists,
                        read_counts=read_counts,
                        haplotypes=haplotypes,
                        ploidy=ploidy,
                        inbreeding=inbreeding,
                        frequencies=prior_frequencies,
                        return_phenotype_prob=True,
                        return_posterior_frequencies=True,
                        return_posterior_occurrence=True,
                    )
                    alleles, _, genotype_prob, phenotype_prob = mode_results[0:4]

                    freqs = mode_results[-2]
                    occur = mode_results[-1]
                    data.sampledata[FORMAT.ACP][sample] = freqs * ploidy
                    data.sampledata[FORMAT.AFP][sample] = freqs
                    data.sampledata[FORMAT.AOP][sample] = occur

                # store variables
                data.sampledata[FORMAT.GT][sample] = alleles
                data.sampledata[FORMAT.GQ][sample] = qual_of_prob(genotype_prob)
                data.sampledata[FORMAT.GPM][sample] = genotype_prob
                data.sampledata[FORMAT.PHPM][sample] = phenotype_prob
                data.sampledata[FORMAT.PHQ][sample] = qual_of_prob(phenotype_prob)
                data.sampledata[FORMAT.MCI][sample] = None
                mec = np.sum(minimum_error_correction(read_calls, haplotypes[alleles]))
                mec_denom = np.sum(read_calls >= 0)
                mecp = mec / mec_denom if mec_denom > 0 else np.nan
                data.sampledata[FORMAT.MEC][sample] = mec
                data.sampledata[FORMAT.MECP][sample] = mecp

            # end of try clause for specific sample
            except Exception as e:
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
                raise SampleAssemblyError(message) from e
        return data
