import sys
import argparse
import numpy as np
from dataclasses import dataclass

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
            With sampledata fields: "read_dists_unique", "read_dist_counts".

        Returns
        -------
        data : LocusAssemblyData
            With columndata fields REF and ALTS, sampledata fields:
            "alleles", "haplotypes", "GQ", "GPM", "PHPM", "PHQ", "MCI"
            and "GL", "GP" if specified and infodata flag "REFMASKED".
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
            "ACP",
            "AFP",
            "AOP",
        ]:
            data.sampledata[field] = dict()
        haplotypes = data.locus.encode_haplotypes()
        mask_reference_allele = data.locus.mask_reference_allele
        prior_frequencies = data.locus.frequencies

        # save allele sequences
        data.columndata["REF"] = data.locus.sequence
        data.columndata["ALTS"] = data.locus.alts
        data.infodata["REFMASKED"] = mask_reference_allele
        data.infodata["AFPRIOR"] = np.round(prior_frequencies, self.precision)

        # check prior frequencies arguments
        if mask_reference_allele:
            assert (prior_frequencies[0] == 0) or np.isnan(prior_frequencies[0])

        # handle invalid scenarios
        # TODO: handle this more elegantly?
        if mask_reference_allele and len(haplotypes) == 1:
            # only allele is masked
            invalid_scenario = True
            data.columndata["FILTER"].append(vcf.filters.NOA.id)
        elif np.any(np.isnan(prior_frequencies)):
            # nan caused by zero freq
            invalid_scenario = True
            data.columndata["FILTER"].append(vcf.filters.AF0.id)
        else:
            invalid_scenario = False

        # mock data for invalid scenario
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
                data.sampledata["ACP"][sample] = np.array([np.nan])
                data.sampledata["AFP"][sample] = np.array([np.nan])
                data.sampledata["AOP"][sample] = np.array([np.nan])
                data.sampledata["GP"][sample] = np.array([np.nan])
                data.sampledata["GL"][sample] = np.array([np.nan])
            return data

        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                ploidy = data.sample_ploidy[sample]
                inbreeding = data.sample_inbreeding[sample]
                reads = data.sampledata["read_dists_unique"][sample]
                read_counts = read_counts = data.sampledata["read_dist_counts"][sample]

                # call haplotypes
                if ("GL" in data.formatfields) or ("GP" in data.formatfields):

                    # calculate full arrays
                    llks = genotype_likelihoods(
                        reads=reads,
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
                        data.sampledata["ACP"][sample] = np.round(
                            counts, self.precision
                        )
                        data.sampledata["AFP"][sample] = np.round(freqs, self.precision)
                        data.sampledata["AOP"][sample] = np.round(occur, self.precision)
                    if "GL" in data.formatfields:
                        data.sampledata["GL"][sample] = np.round(
                            natural_log_to_log10(llks), self.precision
                        )
                    if "GP" in data.formatfields:
                        data.sampledata["GP"][sample] = np.round(
                            probabilities, self.precision
                        )

                else:
                    # use low memory calculation
                    mode_results = posterior_mode(
                        reads=reads,
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

                    freqs = np.round(mode_results[-2], self.precision)
                    occur = np.round(mode_results[-1], self.precision)
                    data.sampledata["ACP"][sample] = np.round(
                        freqs * ploidy, self.precision
                    )
                    data.sampledata["AFP"][sample] = np.round(freqs, self.precision)
                    data.sampledata["AOP"][sample] = np.round(occur, self.precision)

                # store variables
                data.sampledata["alleles"][sample] = alleles
                data.sampledata["haplotypes"][sample] = haplotypes[alleles]
                data.sampledata["GQ"][sample] = qual_of_prob(genotype_prob)
                data.sampledata["GPM"][sample] = np.round(genotype_prob, self.precision)
                data.sampledata["PHPM"][sample] = np.round(
                    phenotype_prob, self.precision
                )
                data.sampledata["PHQ"][sample] = qual_of_prob(phenotype_prob)
                data.sampledata["MCI"][sample] = None

            # end of try clause for specific sample
            except Exception as e:
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
                raise SampleAssemblyError(message) from e
        return data
