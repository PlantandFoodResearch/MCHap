import numpy as np
from numba import njit
from dataclasses import dataclass
from mchap.assemble.classes import Assembler
from mchap.combinatorics import count_unique_genotypes

from mchap.jitutils import seed_numba
from mchap import mset

from .mcmc import mcmc_sampler, greedy_caller
from .utils import posterior_as_array


@dataclass
class CallingMCMC(Assembler):

    ploidy: int
    haplotypes: np.ndarray
    frequencies: np.array = None
    inbreeding: float = 0
    steps: int = 1000
    chains: int = 2
    random_seed: int = None
    step_type: str = "Gibbs"
    """Haplotpye calling using Markov chain Monte Carlo
    for probabilistically encoded variable positions of NGS reads.

    Attributes
    ----------
    ploidy : int
        Ploidy of organism at the assembled locus.
    haplotypes : ndarray, int, shape, (n_haplotypes, n_pos)
        Number of possible alleles at each position in the
        assembled locus.
    frequencies : ndarray, float , shape (n_haplotypes, )
        Optional prior frequencies for each haplotype allele.
    inbreeding : float
        Expected inbreeding coefficient of genotype.
    steps : int, optional
        Number of steps to run in each MCMC simulation
        (default = 1000).
    chains : int, optional
        Number of independent MCMC simulations to run
        (default = 2).
    random_seed : int, optional
        Seed the random seed for numpy and numba RNG
        (default = None).
    step_type : string
        Type of MCMC step to use either "Gibbs" or "Metropolis-Hastings"

    """

    def fit(self, reads, read_counts=None, initial=None):
        """Fit the parametized model to a set of probabilistically
        encoded variable positions of NGS reads.

        Parameters
        ----------
        reads : ndarray, float, shape (n_reads, n_positions, max_allele)
            Probabilistically encoded variable positions of NGS reads.
        read_counts : ndarray, int, shape (n_reads, )
            Optionally specify the number of observations of each read.
        initial : ndarray, int, shape (ploidy, ), optional
            Set the initial genotype state of each MCMC simulation
            (default = None).

        Returns
        -------
        trace : GenotypeMultiTrace
            An instance of GenotypeMultiTrace containing the genotype state
            and log-likelihood at each step in each of the MCMC simulations.

        Notes
        -----
        If the initial genotype state is not set by the user
        then it is automatically set by sampling <ploidy> random
        haplotypes from the mean allele probabilities
        among all reads.
        """
        # handle case of no variants
        if reads.shape[1] == 0:
            # must only have reference allele
            assert len(self.haplotypes) == 1
            genotypes = np.zeros((self.chains, self.steps, self.ploidy), dtype=np.int8)
            llks = np.full((self.chains, self.steps), np.nan)
            return GenotypeAllelesMultiTrace(genotypes, llks, len(self.haplotypes))

        # set random seed once for all chains
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
            seed_numba(self.random_seed)

        # initial genotype
        if initial is None:
            initial = greedy_caller(
                haplotypes=self.haplotypes,
                ploidy=self.ploidy,
                reads=reads,
                read_counts=read_counts,
                inbreeding=self.inbreeding,
            )

        # step type
        if self.step_type == "Gibbs":
            step_type = 0
        elif self.step_type == "Metropolis-Hastings":
            step_type = 1
        else:
            raise ValueError('MCMC step type must be "Gibbs" or "Metropolis-Hastings"')

        genotype_traces = []
        llk_traces = []
        for _ in range(self.chains):
            genotypes, llks = mcmc_sampler(
                genotype_alleles=initial,
                haplotypes=self.haplotypes,
                reads=reads,
                read_counts=read_counts,
                inbreeding=self.inbreeding,
                frequencies=self.frequencies,
                n_steps=self.steps,
                cache=True,
                step_type=step_type,
            )
            genotype_traces.append(genotypes)
            llk_traces.append(llks)
        return GenotypeAllelesMultiTrace(
            np.array(genotype_traces), np.array(llk_traces), len(self.haplotypes)
        )


@dataclass
class GenotypeAllelesMultiTrace(object):
    """Generic class for multi-chain MCMC haplotype assembler trace.

    Attributes
    ----------
    genotypes : ndarray, int, shape (n_chains, n_steps, ploidy)
        The genotype alleles recorded at each step of a Markov chain Monte Carlo
        simulation.
    llks : ndarray, float, shape (n_chains, n_steps)
        The log-likelihood calculated for the genotype at each step of
        the Markov chain Monte Carlo simulation.
    n_allele : int
        Maximum number of alleles
    """

    genotypes: np.ndarray
    llks: np.ndarray
    n_allele: int

    def relabel(self, labels):
        """Returns a new GenotypeTrace object with relabeled alleles.

        Parameters
        ----------
        labels : ndarray, int, shape (n_alleles,)
            New integer labels.

        Returns
        -------
        trace : GenotypeTrace
            A new instance of the GenotypeTrace with relabeled alleles.
        """
        new = type(self)(
            labels[self.genotypes],
            self.llks,
            labels.max() + 1,
        )
        return new

    def burn(self, n):
        """Returns a new GenotypeTrace object without the first
        `n` observations of each chain.

        Parameters
        ----------
        n : int
            Number of steps to remove from the start of each
            chain in the trace.

        Returns
        -------
        trace : GenotypeTrace
            A new instance of the GenotypeTrace without the
            first n steps.

        """
        new = type(self)(
            self.genotypes[:, n:],
            self.llks[:, n:],
            self.n_allele,
        )
        return new

    def posterior(self):
        """Returns a posterior distribution over (phased) genotypes.

        Returns
        -------
        posterior : PosteriorGenotypeDistribution
            A distribution over unique genotypes recorded in the trace.

        """
        # merge chains
        n_chain = self.genotypes.shape[0]
        n_step = self.genotypes.shape[1]
        etc = self.genotypes.shape[2:]
        genotypes = self.genotypes.reshape((n_chain * n_step,) + etc)

        states, counts = mset.unique_counts(genotypes)
        probs = counts / np.sum(counts)

        idx = np.flip(np.argsort(probs))

        return PosteriorGenotypeAllelesDistribution(states[idx], probs[idx])

    def split(self):
        """Split a multitrace into a trace for each component chain.

        Returns
        -------
        traces : iterable
            An iterable of multitraces each containing a single chain.
        """
        for genotypes, llks in zip(self.genotypes, self.llks):
            yield type(self)(
                genotypes[None, ...],
                llks[None, ...],
                self.n_allele,
            )

    def replicate_incongruence(self, threshold=0.6):
        """Identifies incongruence between replicate Markov chains.

        Parameters
        ----------
        threshold : float
            Posterior probability required for a chain to be compaired to others.

        Returns
        -------
        Incongruence : int
            0, 1 or 2 indicating no incongruence, incongruence, or incongruence
            potentially caused by copy-number variation.

        Notes
        -----
        A non-replicated MCMC will always return 0.
        """
        out = 0
        chain_modes = [
            chain.posterior().mode(genotype_support=True) for chain in self.split()
        ]
        alleles = [mode[0] for mode in chain_modes if mode[-1] >= threshold]
        # check for more than one mode
        mode_count = len({array.tobytes() for array in alleles})
        if mode_count > 1:
            out = 1
            # check for more than ploidy alleles
            ploidy = len(alleles[0])
            allele_count = len(set(np.array(alleles).ravel()))
            if allele_count > ploidy:
                out = 2
        return out

    def posterior_frequencies(self):
        """Calculate posterior frequency of haplotype alleles.

        Returns
        -------
        frequencies : ndarray, float, shape (n_alleles, )
            Posterior frequencies of haplotype alleles.
        counts : ndarray, float, shape (n_alleles, )
            Posterior allele counts.
        occurrence : ndarray, float, shape (n_alleles, )
            Posterior probabilities of haplotype occurrence.
        """
        return _posterior_frequencies(self.genotypes, self.n_allele)


@njit
def _posterior_frequencies(genotypes, n_allele):
    n_chain, n_step, ploidy = genotypes.shape
    counts = np.zeros(n_allele)
    occurrence = np.zeros(n_allele)
    for c in range(n_chain):
        for s in range(n_step):
            for i in range(ploidy):
                a = genotypes[c, s, i]
                counts[a] += 1
                first = True
                for j in range(i):
                    b = genotypes[c, s, j]
                    if b == a:
                        first = False
                if first:
                    occurrence[a] += 1
    n_obs = n_chain * n_step
    counts /= n_obs
    occurrence /= n_obs
    return counts / ploidy, counts, occurrence


@dataclass
class PosteriorGenotypeAllelesDistribution(object):
    """Generic class for posterior distribution of
    (finite and countable) genotypes.

    Attributes
    ----------
    genotypes : ndarray, int, shape (n_genotypes, ploidy)
        The possible genotypes of an individual.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probabilities (summing to 1) of each genotype.

    """

    genotypes: np.ndarray
    probabilities: np.ndarray

    def mode(self, genotype_support=False):
        """Return the posterior mode genotype or genotype support.

        Parameters
        ----------
        genotype_support : bool
            If true then the most probable genotype of the mode
            genotype support will be returned in addition with that
            genotypes probability and the mode genotype support probability.

        Returns
        -------
        mode_genotype : ndarray, int, shape (ploidy, n_positions)
            The genotype with the highest posterior probability.
        genotype_probability : float
            The posterior probability of the posterior mode genotype
        genotype_support_probability : float
            The posterior probability of the posterior mode genotype

        """
        if genotype_support is False:
            idx = np.argmax(self.probabilities)
            return self.genotypes[idx], self.probabilities[idx]
        else:
            labels = np.zeros(len(self.genotypes), dtype=int)
            support_labels = {}  # string: int
            probs = {}  # int: float

            for i, gen in enumerate(self.genotypes):
                genotype_support = mset.unique(gen)
                string = genotype_support.tobytes()
                if string not in support_labels:
                    label = i
                    support_labels[string] = label
                    probs[label] = self.probabilities[i]
                else:
                    label = support_labels[string]
                    probs[label] += self.probabilities[i]
                labels[i] = label

            support_labels, probs = zip(*probs.items())
            mode = support_labels[np.argmax(probs)]
            idx = labels == mode
            genotypes = self.genotypes[idx]
            probs = self.probabilities[idx]
            idx = np.argmax(probs)
            return genotypes[idx], probs[idx], probs.sum()

    def as_array(self, n_alleles):
        _, ploidy = self.genotypes.shape
        u_genotypes = count_unique_genotypes(n_alleles, ploidy)
        return posterior_as_array(self.genotypes, self.probabilities, u_genotypes)
