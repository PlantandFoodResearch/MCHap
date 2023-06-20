import numpy as np
from numba import njit
from dataclasses import dataclass

from mchap.assemble.classes import Assembler
from mchap.jitutils import seed_numba
from mchap.calling.mcmc import greedy_caller
from mchap.calling.classes import GenotypeAllelesMultiTrace

from .mcmc import mcmc_sampler
from .validation import duo_valid, trio_valid


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
    step_type: str = "Gibbs"

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


        # step type
        if self.step_type == "Gibbs":
            step_type = 0
        elif self.step_type == "Metropolis-Hastings":
            step_type = 1
        else:
            raise ValueError('MCMC step type must be "Gibbs" or "Metropolis-Hastings"')

        shape = (self.chains, self.steps, n_samples, max_ploidy)
        trace = np.empty(shape=shape, dtype=np.int16)
        for i in range(self.chains):
            trace[i] = mcmc_sampler(
                sample_genotypes=initial,
                sample_ploidy=self.sample_ploidy,
                sample_parents=self.sample_parents,
                gamete_tau=self.gamete_tau,
                gamete_lambda=self.gamete_lambda,
                gamete_error=self.gamete_error,
                sample_read_dists=sample_reads,
                sample_read_counts=sample_read_counts,
                haplotypes=self.haplotypes,
                n_steps=self.steps,
                annealing=self.annealing,
                step_type=step_type,
            )
        return PedigreeAllelesMultiTrace(trace)


@njit(cache=True)
def _trace_incongruence(
    trace, sample_ploidy, sample_parents, gamete_tau, gamete_lambda
):
    n_obs, n_samples, _ = trace.shape
    assert sample_ploidy.shape == (n_samples,)
    assert sample_parents.shape == (n_samples, 2)
    assert gamete_tau.shape == (n_samples, 2)
    assert gamete_lambda.shape == (n_samples, 2)
    out = np.zeros(n_samples)

    for o in range(n_obs):
        for i in range(n_samples):
            p, q = sample_parents[i, 0], sample_parents[i, 1]
            if (p < 0) and (q < 0):
                valid = True
            elif p < 0:
                valid = duo_valid(
                    progeny=trace[o, i][0 : sample_ploidy[i]],
                    parent=trace[o, q][0 : sample_ploidy[q]],
                    tau=gamete_tau[i, 1],
                    lambda_=gamete_lambda[i, 1],
                )
            elif q < 0:
                valid = duo_valid(
                    progeny=trace[o, i][0 : sample_ploidy[i]],
                    parent=trace[o, p][0 : sample_ploidy[p]],
                    tau=gamete_tau[i, 0],
                    lambda_=gamete_lambda[i, 0],
                )
            else:
                valid = trio_valid(
                    progeny=trace[o, i][0 : sample_ploidy[i]],
                    parent_p=trace[o, p][0 : sample_ploidy[p]],
                    parent_q=trace[o, q][0 : sample_ploidy[q]],
                    tau_p=gamete_tau[i, 0],
                    tau_q=gamete_tau[i, 1],
                    lambda_p=gamete_lambda[i, 0],
                    lambda_q=gamete_lambda[i, 1],
                )
            if not valid:
                out[i] += 1
    out /= n_obs
    return out


@dataclass
class PedigreeAllelesMultiTrace(object):
    genotypes: np.ndarray

    def burn(self, n):
        new = type(self)(self.genotypes[:, n:])
        return new

    def individual(self, index):
        sample_trace = self.genotypes[:, :, index, :]
        ploidy = (sample_trace[0, 0] >= 0).sum()
        return GenotypeAllelesMultiTrace(
            sample_trace[:, :, 0:ploidy],
            np.full(self.genotypes.shape[0:2], np.nan),
        )

    def incongruence(self, sample_ploidy, sample_parents, gamete_tau, gamete_lambda):
        trace = self.genotypes
        n_chains, n_steps, n_samples, max_ploidy = trace.shape
        trace = trace.reshape(n_chains * n_steps, n_samples, max_ploidy)
        return _trace_incongruence(
            trace, sample_ploidy, sample_parents, gamete_tau, gamete_lambda
        )
