import numpy as np
from dataclasses import dataclass
from mchap.assemble.classes import Assembler

from mchap.jitutils import seed_numba
from mchap.calling.mcmc import greedy_caller
from mchap.calling.classes import GenotypeAllelesMultiTrace

from .mcmc import mcmc_sampler


@dataclass
class PedigreeCallingMCMC(Assembler):
    sample_ploidy: np.ndarray
    sample_inbreeding: np.ndarray
    sample_parents: np.ndarray
    gamete_tau: np.ndarray
    gamete_lambda: np.ndarray
    gamete_error: np.ndarray
    haplotypes: np.ndarray
    steps: int = 2000
    annealing: int = 1000
    chains: int = 2
    random_seed: int = None

    def fit(self, sample_reads, sample_read_counts, initial=None):
        n_samples = len(self.sample_ploidy)
        max_ploidy = self.sample_ploidy.max()

        # set random seed once for all chains
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            seed_numba(self.random_seed)

        if initial is None:
            initial = np.full((n_samples, max_ploidy), -1, np.int16)
            for i in range(n_samples):
                genotype = greedy_caller(
                    haplotypes=self.haplotypes,
                    ploidy=self.sample_ploidy[i],
                    reads=sample_reads[i],
                    read_counts=sample_read_counts[i],
                    inbreeding=self.sample_inbreeding[i],
                )
                initial[i][0 : self.sample_ploidy[i]] = genotype

        shape = (self.chains, self.steps, n_samples, max_ploidy)
        trace = np.empty(shape=shape, dtype=np.int16)
        for i in range(self.chains):
            trace[i] = mcmc_sampler(
                sample_genotypes=initial,
                sample_ploidy=self.sample_ploidy,
                sample_inbreeding=self.sample_inbreeding,
                sample_parents=self.sample_parents,
                gamete_tau=self.gamete_tau,
                gamete_lambda=self.gamete_lambda,
                gamete_error=self.gamete_error,
                sample_read_dists=sample_reads,
                sample_read_counts=sample_read_counts,
                haplotypes=self.haplotypes,
                n_steps=self.steps,
                annealing=self.annealing,
            )
        trace = np.sort(trace, axis=-1)
        return PedigreeAllelesMultiTrace(trace, self.sample_ploidy)


@dataclass
class PedigreeAllelesMultiTrace(object):
    genotypes: np.ndarray
    ploidy: np.ndarray

    def burn(self, n):
        new = type(self)(self.genotypes[:, n:], self.ploidy)
        return new

    def individual(self, index):
        trace = self.genotypes[:, :, index, :]
        ploidy = self.ploidy[index]
        if ploidy < trace.shape[-1]:
            # need to remove padding
            trace = np.sort(trace, axis=-1)
            trace = trace[..., ploidy:]
        return GenotypeAllelesMultiTrace(
            self.genotypes[:, :, index, :],
            np.full(self.genotypes.shape[0:2], np.nan),
        )
