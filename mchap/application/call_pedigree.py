import sys
import argparse
import numpy as np
from dataclasses import dataclass

from mchap.application import call_baseclass
from mchap.application.baseclass import SampleAssemblyError, SAMPLE_ASSEMBLY_ERROR
from mchap.application.arguments import (
    CALL_PEDIGREE_MCMC_PARSER_ARGUMENTS,
    collect_call_pedigree_mcmc_program_arguments,
)
from mchap.pedigree.classes import PedigreeCallingMCMC
from mchap.calling.exact import genotype_likelihoods
from mchap.jitutils import natural_log_to_log10

from mchap.io import qual_of_prob, vcf


@dataclass
class program(call_baseclass.program):
    sample_parents: dict = None
    gamete_ploidy: dict = None
    gamete_ibd: dict = None
    gamete_error: dict = None
    mcmc_chains: int = 1
    mcmc_steps: int = 2000
    mcmc_burn: int = 1000
    mcmc_incongruence_threshold: float = 0.60

    @classmethod
    def cli(cls, command):
        """Program initialization from cli command

        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser(
            "MCMC haplotype calling via pedigree-annealing. "
            "WARNING THIS TOOL IS HIGHLY EXPERIMENTAL!!!"
        )
        for arg in CALL_PEDIGREE_MCMC_PARSER_ARGUMENTS:
            arg.add_to(parser)

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])

        # sort argument details
        arguments = collect_call_pedigree_mcmc_program_arguments(args)
        return cls(cli_command=command, **arguments)

    def call_sample_genotypes(self, data):
        """Pedigree based genotype calling.

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
            "AFP",
        ]:
            data.sampledata[field] = dict()
        # get haplotypes and metadata
        haplotypes = data.locus.encode_haplotypes()
        haplotype_frequencies = data.locus.frequencies
        mask_reference_allele = data.locus.mask_reference_allele
        mask = np.zeros(len(haplotypes), bool)
        mask[0] = mask_reference_allele

        # save allele sequences
        data.columndata["REF"] = data.locus.sequence
        data.columndata["ALTS"] = data.locus.alts
        data.infodata["REFMASKED"] = mask_reference_allele
        data.infodata["AFPRIOR"] = np.round(haplotype_frequencies, self.precision)

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
        else:
            prior_frequencies = None

        # handle invalid scenarios
        # TODO: handle this more elegantly?
        if len(mcmc_haplotypes) == 0:
            # must have one or more haplotypes for MCMC
            invalid_scenario = True
            data.columndata["FILTER"].append(vcf.filters.NOA.id)
        elif self.use_haplotype_frequencies_prior and np.any(
            np.isnan(prior_frequencies)
        ):
            # nan caused by zero freq
            invalid_scenario = True
            data.columndata["FILTER"].append(vcf.filters.AF0.id)
        else:
            invalid_scenario = False
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

        # combine samples reads into a single array
        n_samples = len(data.samples)
        max_reads = max(
            len(data.sampledata["read_dists_unique"][s]) for s in data.samples
        )
        n_pos = len(data.locus.positions)
        max_nucl = max([len(a) for a in data.locus.alleles] + [0])
        sample_reads = np.full((n_samples, max_reads, n_pos, max_nucl), np.nan)
        sample_read_counts = np.zeros((n_samples, max_reads), np.int64)
        for i, sample in enumerate(data.samples):
            _reads = data.sampledata["read_dists_unique"][sample]
            _counts = data.sampledata["read_dist_counts"][sample]
            assert len(_reads) == len(_counts)
            sample_reads[i, 0 : len(_reads)] = _reads
            sample_read_counts[i, 0 : len(_counts)] = _counts

        # convert pedigree data to arrays, TODO: do this once when parsing arguments?
        pedigree_position = {s: i for i, s in enumerate(data.samples)}
        pedigree_position[None] = -1
        n_samples = len(data.samples)
        parent_indices = np.full((n_samples, 2), -1, dtype=int)
        gamete_tau = np.full((n_samples, 2), -1, dtype=int)
        gamete_lambda = np.full((n_samples, 2), np.nan, dtype=float)
        gamete_error = np.full((n_samples, 2), np.nan, dtype=float)
        for i, s in enumerate(data.samples):
            for j, p in enumerate(self.sample_parents[s]):
                try:
                    parent_indices[i, j] = pedigree_position[p]
                except KeyError as e:
                    raise KeyError(
                        "Parent identifier '{}' is not a sample identifier".format(p)
                    ) from e
            gamete_tau[i] = self.gamete_ploidy[s]
            gamete_lambda[i] = self.gamete_ibd[s]
            gamete_error[i] = self.gamete_error[s]

        # pedigree based assembly
        pedigree_trace = (
            PedigreeCallingMCMC(
                sample_ploidy=np.array([data.sample_ploidy[s] for s in data.samples]),
                sample_inbreeding=np.array(
                    [data.sample_inbreeding[s] for s in data.samples]
                ),
                sample_parents=parent_indices,
                gamete_tau=gamete_tau,
                gamete_lambda=gamete_lambda,
                gamete_error=gamete_error,
                haplotypes=mcmc_haplotypes,
                steps=self.mcmc_steps,
                annealing=self.mcmc_burn,  # anneal over full burn-in period
                chains=self.mcmc_chains,
                random_seed=self.random_seed,
            )
            .fit(
                sample_reads=sample_reads,
                sample_read_counts=sample_read_counts,
            )
            .burn(self.mcmc_burn)
        )  # burn all annealing

        # iterate of samples to summarize
        for i, sample in enumerate(data.samples):
            # wrap in try clause to pass sample info back with any exception
            try:
                trace = pedigree_trace.individual(i)
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
                        reads=data.sampledata["read_dists_unique"][sample],
                        read_counts=data.sampledata["read_dist_counts"][sample],
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
