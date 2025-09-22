import sys
import argparse
import numpy as np
from dataclasses import dataclass
import pysam

import mchap.io.vcf.infofields as INFO
import mchap.io.vcf.formatfields as FORMAT
import mchap.io.vcf.columns as COLUMN
from mchap import mset
from mchap.application import baseclass
from mchap.application.baseclass import SampleAssemblyError, SAMPLE_ASSEMBLY_ERROR
from mchap.encoding.integer import minimum_error_correction

from mchap import combinatorics
from mchap.assemble import (
    DenovoMCMC,
    call_posterior_haplotypes,
)
from mchap.calling.exact import genotype_likelihoods
from mchap.jitutils import (
    natural_log_to_log10,
    genotype_alleles_as_index,
)
from mchap.io import (
    Locus,
    read_bed4,
    qual_of_prob,
    vcf,
)

from .arguments import (
    ASSEMBLE_MCMC_PARSER_ARGUMENTS,
    collect_assemble_mcmc_program_arguments,
)


@dataclass
class program(baseclass.program):
    bed: str = ""
    region: str = None
    region_id: str = None
    haplotype_posterior_threshold: float = 0.2
    mcmc_chains: int = 1
    mcmc_steps: int = 2000
    mcmc_burn: int = 1000
    mcmc_alpha: float = 1.0
    mcmc_beta: float = 3.0
    mcmc_fix_homozygous: float = 0.999
    mcmc_recombination_step_probability: float = 0.5
    mcmc_partial_dosage_step_probability: float = 0.5
    mcmc_dosage_step_probability: bool = 1.0
    mcmc_incongruence_threshold: float = 0.60
    mcmc_llk_cache_threshold: int = 100
    sample_mcmc_temperatures: dict = None

    @classmethod
    def cli(cls, command):
        """Program initialization from cli command

        e.g. `program.cli(sys.argv)`
        """
        parser = argparse.ArgumentParser("MCMC haplotype assembly")
        for arg in ASSEMBLE_MCMC_PARSER_ARGUMENTS:
            arg.add_to(parser)

        if len(command) < 3:
            parser.print_help()
            sys.exit(1)
        args = parser.parse_args(command[2:])

        # sort argument details
        arguments = collect_assemble_mcmc_program_arguments(args)
        return cls(cli_command=command, **arguments)

    def loci(self):
        if (self.bed is None) and (self.region is None):
            raise ValueError("No region or targets bedfile is specified.")
        elif self.bed is not None:
            bed = read_bed4(self.bed)
            for b in bed:
                yield b.set_sequence(self.ref).set_variants(self.vcf)
        else:
            locus = Locus.from_region_string(self.region, self.region_id)
            yield locus.set_sequence(self.ref).set_variants(self.vcf)

    def header_contigs(self):
        with pysam.Fastafile(self.ref) as fasta:
            contigs = [
                vcf.headermeta.ContigHeader(c, l)
                for c, l in zip(fasta.references, fasta.lengths)
            ]
        return contigs

    def call_sample_genotypes(self, data):
        """De novo haplotype assembly of each sample.

        Parameters
        ----------
        data : LocusAssemblyData

        Returns
        -------
        data : LocusAssemblyData
        """
        # dict to temporarily store posteriors
        sample_modes = dict()
        sample_posteriors = dict()
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                # prior
                if data.sample_inbreeding is None:
                    # flat prior over genotypes
                    inbreeding = None
                else:
                    # Dirichlet-multinomial with flat prior of allele frequencies
                    inbreeding = data.sample_inbreeding[sample]
                # assembly MCMC
                read_calls = data.read_calls[sample]
                read_dists = data.read_dists[sample]
                read_counts = data.read_counts[sample]
                trace = (
                    DenovoMCMC(
                        ploidy=data.sample_ploidy[sample],
                        n_alleles=data.locus.count_alleles(),
                        inbreeding=inbreeding,
                        steps=self.mcmc_steps,
                        chains=self.mcmc_chains,
                        fix_homozygous=self.mcmc_fix_homozygous,
                        recombination_step_probability=self.mcmc_recombination_step_probability,
                        partial_dosage_step_probability=self.mcmc_partial_dosage_step_probability,
                        dosage_step_probability=self.mcmc_dosage_step_probability,
                        temperatures=self.sample_mcmc_temperatures[sample],
                        random_seed=self.random_seed,
                        llk_cache_threshold=self.mcmc_llk_cache_threshold,
                    )
                    .fit(
                        reads=read_dists,
                        read_counts=read_counts,
                    )
                    .burn(self.mcmc_burn)
                )
                posterior = trace.posterior()
                sample_posteriors[sample] = posterior

                # genotype support results
                genotype_support = posterior.mode_genotype_support()
                genotype_support_prob = genotype_support.probabilities.sum()
                data.sampledata[FORMAT.SPM][sample] = genotype_support_prob
                data.sampledata[FORMAT.SQ][sample] = qual_of_prob(genotype_support_prob)

                # genotype results
                genotype, genotype_prob = genotype_support.mode_genotype()
                sample_modes[sample] = genotype
                data.sampledata[FORMAT.GQ][sample] = qual_of_prob(genotype_prob)
                data.sampledata[FORMAT.GPM][sample] = genotype_prob

                # MEC
                mec = np.sum(minimum_error_correction(read_calls, genotype))
                mec_denom = np.sum(read_calls >= 0)
                mecp = mec / mec_denom if mec_denom > 0 else np.nan
                data.sampledata[FORMAT.MEC][sample] = mec
                data.sampledata[FORMAT.MECP][sample] = mecp

                # chain incongruence
                incongruence = trace.replicate_incongruence(
                    threshold=self.mcmc_incongruence_threshold
                )
                data.sampledata[FORMAT.MCI][sample] = incongruence

            # end of try clause for specific sample
            except Exception as e:
                path = data.sample_bams.get(sample)
                message = SAMPLE_ASSEMBLY_ERROR.format(sample=sample, bam=path)
                raise SampleAssemblyError(message) from e

        # call posterior haplotypes and sort and map to allele numbers
        haplotypes, ref_called = call_posterior_haplotypes(
            list(sample_posteriors.values()),
            threshold=self.haplotype_posterior_threshold,
        )
        haplotype_labels = {h.tobytes(): i for i, h in enumerate(haplotypes)}

        # record if reference allele was not observed
        data.infodata[INFO.REFMASKED] = not ref_called
        if not ref_called:
            # remove from labeling
            haplotype_labels.pop(haplotypes[0].tobytes())
            # filter for no observed alleles
            if len(haplotypes) == 1:
                data.columndata[COLUMN.FILTER].append(vcf.filters.NOA.id)

        # decode and save alt alleles
        if len(haplotypes) > 1:
            alts = data.locus.format_haplotypes(haplotypes[1:])
        else:
            alts = []
        data.columndata[COLUMN.REF] = data.locus.sequence
        data.columndata[COLUMN.ALT] = alts

        # encode sample haplotypes as alleles
        for sample in data.samples:
            # wrap in try clause to pass sample info back with any exception
            try:
                # encode sample haplotypes as alleles
                alleles = _genotype_as_alleles(
                    sample_modes[sample],
                    haplotype_labels,
                )
                data.sampledata[FORMAT.GT][sample] = alleles

                # posterior allele frequencies/occurrences if requested
                if self.require_AFP():
                    frequencies = np.zeros(len(haplotypes))
                    occurrences = np.zeros(len(haplotypes))
                    haps, freqs, occur = sample_posteriors[sample].allele_frequencies()
                    idx = mset.categorize(haplotypes, haps)
                    frequencies[idx >= 0] = freqs[idx[idx >= 0]]
                    occurrences[idx >= 0] = occur[idx[idx >= 0]]
                    data.sampledata[FORMAT.AFP][sample] = frequencies
                    data.sampledata[FORMAT.AOP][sample] = occurrences
                    data.sampledata[FORMAT.ACP][sample] = (
                        frequencies * data.sample_ploidy[sample]
                    )

                # encode posterior probabilities if requested
                if FORMAT.GP in data.formatfields:
                    probabilities = _genotype_posterior_as_array(
                        sample_posteriors[sample],
                        haplotype_labels,
                    )
                    data.sampledata[FORMAT.GP][sample] = probabilities
                # genotype likelihoods if requested
                if FORMAT.GL in data.formatfields:
                    read_dists = data.read_dists[sample]
                    read_counts = data.read_counts[sample]
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


def _genotype_as_alleles(genotype, labels):
    """Convert a  genotype of haplotype arrays to an array
    of VCF sorted allele integers.
    Parameters
    ----------
    genotype : ndarray, int, shape (ploidy, n_positions)
        Integer encoded genotype.
    labels : dict[bytes, int]
        Map of haplotype bytes to allele number e.g.
        `{h.tobytes(): i for i, h in enumerate(haplotypes)}`.

    Returns
    -------
    alleles : ndarray, int, shape (ploidy, )
        VCF sorted alleles.
    """
    alleles = np.sort([labels.get(h.tobytes(), -1) for h in genotype])
    alleles = np.append(alleles[alleles >= 0], alleles[alleles < 0])
    return alleles


def _genotype_posterior_as_array(posterior, labels):
    """Convert a  genotype of haplotype arrays to an array
    of VCF sorted allele integers.
    Parameters
    ----------
    posterior : PosteriorGenotypeDistribution
        Posterior genotype distribution.
    labels : dict[bytes, int]
        Map of haplotype bytes to allele number e.g.
        `{h.tobytes(): i for i, h in enumerate(haplotypes)}`.

    Returns
    -------
    probabilities : ndarray, float, shape (unique_genotypes, )
        Probabilities of all possible genotypes given labeled
        alleles.
    """
    n_alleles = len(labels)
    _, ploidy, _ = posterior.genotypes.shape
    u_gens = combinatorics.count_unique_genotypes(n_alleles, ploidy)
    probabilities = np.zeros(u_gens, float)
    for haps, prob in zip(posterior.genotypes, posterior.probabilities):
        alleles = np.sort([labels.get(h.tobytes(), -1) for h in haps])
        if alleles[0] < 0:
            # contains non-called allele so cannot be encoded
            pass
        else:
            idx = genotype_alleles_as_index(alleles)
            probabilities[idx] = prob
    return probabilities
