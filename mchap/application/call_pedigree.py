import sys
import argparse
import numpy as np
from dataclasses import dataclass
import warnings

import mchap.io.vcf.infofields as INFO
import mchap.io.vcf.formatfields as FORMAT
import mchap.io.vcf.columns as COLUMN
from mchap.application import call_baseclass
from mchap.application.baseclass import SampleAssemblyError, SAMPLE_ASSEMBLY_ERROR
from mchap.application.arguments import (
    CALL_PEDIGREE_MCMC_PARSER_ARGUMENTS,
    collect_call_pedigree_mcmc_program_arguments,
)
from mchap.pedigree.classes import PedigreeCallingMCMC
from mchap.calling.exact import genotype_likelihoods
from mchap.jitutils import natural_log_to_log10

from mchap.encoding.integer import minimum_error_correction
from mchap.io import qual_of_prob, vcf


class ExperimentalFeatureWarning(UserWarning):
    pass


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
        warnings.warn(
            "THIS PROGRAM IS HIGHLY EXPERIMENTAL!!!", ExperimentalFeatureWarning
        )
        parser = argparse.ArgumentParser(
            "MCMC haplotype calling via pedigree-annealing. "
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

        # mask zero frequency haplotypes if using prior
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

        # combine samples reads into a single array
        n_samples = len(data.samples)
        max_reads = max(len(data.read_dists[s]) for s in data.samples)
        n_pos = len(data.locus.positions)
        max_nucl = max([len(a) for a in data.locus.alleles] + [0])
        sample_reads = np.full((n_samples, max_reads, n_pos, max_nucl), np.nan)
        sample_read_counts = np.zeros((n_samples, max_reads), np.int64)
        for i, sample in enumerate(data.samples):
            _reads = data.read_dists[sample]
            _counts = data.read_counts[sample]
            assert len(_reads) == len(_counts)
            sample_reads[i, 0 : len(_reads)] = _reads
            sample_read_counts[i, 0 : len(_counts)] = _counts

        # convert pedigree data to arrays, TODO: do this once when parsing arguments?
        pedigree_position = {s: i for i, s in enumerate(data.samples)}
        pedigree_position[None] = -1
        n_samples = len(data.samples)
        sample_ploidy = np.array([data.sample_ploidy[s] for s in data.samples])
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
                sample_ploidy=sample_ploidy,
                sample_inbreeding=np.array(
                    [data.sample_inbreeding[s] for s in data.samples]
                ),
                sample_parents=parent_indices,
                gamete_tau=gamete_tau,
                gamete_lambda=gamete_lambda,
                gamete_error=gamete_error,
                haplotypes=mcmc_haplotypes,
                frequencies=mcmc_prior_frequencies,
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
        pedigree_posterior_error = pedigree_trace.incongruence(
            sample_ploidy=sample_ploidy,
            sample_parents=parent_indices,
            gamete_tau=gamete_tau,
            gamete_lambda=gamete_lambda,
        )

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
                alleles, genotype_prob, support_prob = posterior.mode(
                    genotype_support=True
                )

                # store variables
                data.sampledata[FORMAT.GT][sample] = alleles
                # data.sampledata["haplotypes"][sample] = haplotypes[alleles]
                data.sampledata[FORMAT.GQ][sample] = qual_of_prob(genotype_prob)
                data.sampledata[FORMAT.GPM][sample] = genotype_prob
                data.sampledata[FORMAT.SPM][sample] = support_prob
                data.sampledata[FORMAT.SQ][sample] = qual_of_prob(support_prob)
                data.sampledata[FORMAT.MCI][sample] = incongruence
                data.sampledata[FORMAT.PEDERR][sample] = pedigree_posterior_error[i]
                _read_calls = data.read_calls[sample]
                mec = np.sum(minimum_error_correction(_read_calls, haplotypes[alleles]))
                mec_denom = np.sum(_read_calls >= 0)
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
                        reads=data.read_dists[sample],
                        read_counts=data.read_counts[sample],
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
