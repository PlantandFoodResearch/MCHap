import sys
import argparse
import numpy as np
from dataclasses import dataclass

from mchap.application import call_baseclass
from mchap.application.baseclass import SampleAssemblyError, SAMPLE_ASSEMBLY_ERROR
from mchap.application.arguments import (
    CALL_MCMC_PARSER_ARGUMENTS,
    collect_call_mcmc_program_arguments,
)
from mchap.calling.classes import CallingMCMC
from mchap.calling.exact import genotype_likelihoods
from mchap.jitutils import natural_log_to_log10

from mchap.io import qual_of_prob


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
            With sampledata fields: "read_dists_unique", "read_dist_counts".

        Returns
        -------
        data : LocusAssemblyData
            With sampledata fields: "alleles", "haplotypes", "GQ", "GPM", "PHPM", "PHQ", "MCI"
            and "GL", "GP" if specified and infodata flag "NRO".
        """
        for field in [
            "alleles",
            "haplotypes",
            "GQ",
            "GPM",
            "PHPM",
            "PHQ",
            "MCI",
            "GL",
            "GP",
            "AFP",
        ]:
            data.sampledata[field] = dict()
        # get haplotypes and metadata
        haplotypes = data.locus.encode_haplotypes()
        haplotype_frequencies = data.locus.frequencies
        mask_reference_allele = data.locus.mask_reference_allele
        mask = np.zeros(len(haplotypes), bool)
        mask[0] = mask_reference_allele

        # record if reference allele was masked
        data.infodata["REFMASKED"] = mask_reference_allele

        # mask zero frequency haplotypes if using prior
        if self.use_haplotype_frequencies_prior:
            mask |= haplotype_frequencies == 0

        # remove masked haplotypes from mcmc
        if np.any(mask):
            mcmc_haplotypes = haplotypes[~mask]
            mcmc_haplotype_labels = np.where(~mask)[0]
        else:
            # use all haplotypes
            mcmc_haplotype_labels = None
            mcmc_haplotypes = haplotypes

        # must have one or more haplotypes for MCMC
        invalid_scenario = len(mcmc_haplotypes) == 0

        # get prior for allele frequencies
        if self.use_haplotype_frequencies_prior:
            prior_frequencies = haplotype_frequencies[~mask]
            invalid_scenario |= np.any(np.isnan(prior_frequencies))
        else:
            prior_frequencies = None

        # need to mock null results if we can do MCMC
        # TODO: handle this more elegantly?
        if invalid_scenario:
            for sample in data.samples:
                ploidy = data.sample_ploidy[sample]
                data.sampledata["alleles"][sample] = np.full(ploidy, -1, int)
                data.sampledata["haplotypes"][sample] = np.full(
                    (ploidy, len(haplotypes[0])), -1, int
                )
                data.sampledata["GQ"][sample] = np.nan
                data.sampledata["GPM"][sample] = np.nan
                data.sampledata["PHPM"][sample] = np.nan
                data.sampledata["PHQ"][sample] = np.nan
                data.sampledata["MCI"][sample] = np.nan
                data.sampledata["AFP"][sample] = np.array([np.nan])
                data.sampledata["GP"][sample] = np.array([np.nan])
                data.sampledata["GL"][sample] = np.array([np.nan])
            return data

        # iterate of samples
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                reads = data.sampledata["read_dists_unique"][sample]
                read_counts = read_counts = data.sampledata["read_dist_counts"][sample]
                # call haplotypes
                trace = (
                    CallingMCMC(
                        ploidy=data.sample_ploidy[sample],
                        haplotypes=mcmc_haplotypes,
                        inbreeding=data.sample_inbreeding[sample],
                        frequencies=prior_frequencies,
                        steps=self.mcmc_steps,
                        chains=self.mcmc_chains,
                        random_seed=self.random_seed,
                    )
                    .fit(
                        reads=reads,
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
                alleles, genotype_prob, phenotype_prob = posterior.mode(phenotype=True)

                # store variables
                data.sampledata["alleles"][sample] = alleles
                data.sampledata["haplotypes"][sample] = haplotypes[alleles]
                data.sampledata["GQ"][sample] = qual_of_prob(genotype_prob)
                data.sampledata["GPM"][sample] = np.round(genotype_prob, self.precision)
                data.sampledata["PHPM"][sample] = np.round(
                    phenotype_prob, self.precision
                )
                data.sampledata["PHQ"][sample] = qual_of_prob(phenotype_prob)
                data.sampledata["MCI"][sample] = incongruence

                # posterior allele frequencies if requested
                if "AFP" in data.formatfields:
                    frequencies = np.zeros(len(haplotypes))
                    alleles, counts = np.unique(trace.genotypes, return_counts=True)
                    frequencies[alleles] = counts / counts.sum()
                    data.sampledata["AFP"][sample] = np.round(
                        frequencies, self.precision
                    )

                # genotype posteriors if requested
                if "GP" in data.formatfields:
                    probabilities = posterior.as_array(len(haplotypes))
                    data.sampledata["GP"][sample] = np.round(
                        probabilities, self.precision
                    )

                # genotype likelihoods if requested
                if "GL" in data.formatfields:
                    llks = genotype_likelihoods(
                        reads=reads,
                        read_counts=read_counts,
                        ploidy=data.sample_ploidy[sample],
                        haplotypes=haplotypes,
                    )
                    data.sampledata["GL"][sample] = np.round(
                        natural_log_to_log10(llks), self.precision
                    )

            # end of try clause for specific sample
            except Exception as e:
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
                raise SampleAssemblyError(message) from e
        return data
