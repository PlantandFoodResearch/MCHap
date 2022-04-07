import sys
import argparse
import numpy as np
from dataclasses import dataclass

from mchap.application import baseclass
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
from mchap.jitutils import natural_log_to_log10, index_as_genotype_alleles

from mchap.io import qual_of_prob


@dataclass
class program(baseclass.program):
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
            With sampledata fields: "alleles", "haplotypes", "GQ", "GPM", "PHPM", "PHQ", "MCI"
            and "GL", "GP" if specified.
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
        haplotypes = data.locus.encode_haplotypes()
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                ploidy = data.sample_ploidy[sample]
                inbreeding = data.sample_inbreeding[sample]
                reads = data.sampledata["read_dists_unique"][sample]
                read_counts = read_counts = data.sampledata["read_dist_counts"][sample]

                # call haplotypes
                if (
                    ("GL" in data.formatfields)
                    or ("GP" in data.formatfields)
                    or ("AFP" in data.formatfields)
                ):
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
                    )
                    idx = np.argmax(probabilities)
                    alleles = index_as_genotype_alleles(idx, ploidy)
                    genotype_prob = probabilities[idx]
                    _, phenotype_probs = alternate_dosage_posteriors(
                        alleles, probabilities
                    )
                    phenotype_prob = phenotype_probs.sum()

                    # store specified arrays
                    if "AFP" in data.formatfields:
                        freqs = posterior_allele_frequencies(
                            probabilities, ploidy, len(haplotypes)
                        )
                        data.sampledata["AFP"][sample] = np.round(freqs, self.precision)
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
                    alleles, _, genotype_prob, phenotype_prob = posterior_mode(
                        reads=reads,
                        read_counts=read_counts,
                        haplotypes=haplotypes,
                        ploidy=ploidy,
                        inbreeding=inbreeding,
                        return_phenotype_prob=True,
                    )

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
