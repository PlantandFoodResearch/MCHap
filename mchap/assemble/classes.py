import numpy as np
from dataclasses import dataclass

from mchap import mset
from mchap.encoding import allelic


@dataclass
class Assembler(object):
    """Abstract base class for haplotype assemblers.

    """
            
    @classmethod
    def parameterize(cls, *args, **kwargs):
        """Returns an instance with specified parameters.

        Parameters
        ----------
        args : list
            Positional paramaters for the model.
        kwargs : dict
            Key-word parameters for the model.

        Returns
        -------
        assembler : Assembler
            An instance of class Assembler.

        Notes
        -----
        This is conveniance method to create a delayed instance of a 
        class with Dask.

        """
        return cls(*args, **kwargs)


    def fit(self):
        """Fit an assembler model to a dataset.

        Raises
        ------
        NotImplementedError

        """
        raise NotImplementedError()


@dataclass
class PosteriorGenotypeDistribution(object):
    """Generic class for posterior distribution of 
    (finite and countable) genotypes.

    Attributes
    ----------
    genotypes : ndarray, int, shape (n_genotypes, ploidy, n_positions)
        The possible genotypes of an individual of known ploidy at a 
        locus covering n_positions variable positions.
    probabilities : ndarray, float, shape (n_genotypes, )
        Probabilities (summing to 1) of each genotype.

    """

    genotypes: np.ndarray
    probabilities: np.ndarray

    def mode(self):
        """Return the posterior mode genotype.

        Returns
        -------
        mode_genotype : ndarray, int, shape (ploidy, n_positions)
            The genotype with the highest posterior probability.
        probability : float
            The posterior probability of the posterior mode genotype

        """
        idx = np.argmax(self.probabilities)
        return self.genotypes[idx], self.probabilities[idx]

    def mode_phenotype(self):
        """Return genotypes congruent with the posterior mode phenotype.

        Returns
        -------
        mode_genotypes : ndarray, int, shape (n_genotypes, ploidy, n_positions)
            Genotypes which contain only the haplotypes found in the posterior
            mode phenotype (at variable dosage levels).
        probabilities : ndarray, float, shape (n_genotypes, )
            The posterior probabilities asociated with each 
            genotype which is congruent with the posterior 
            mode phenotype.

        Notes
        -----
        The term 'phenotype' is used here to describe a set of unique haplotypes
        without dosage information.
        Hence for a given phenotype and ploidy > 2 their are one or more congruent 
        genotypes that consist of the haplotypes of that genotype.

        """
        labels = np.zeros(len(self.genotypes), dtype=np.int)
        phenotype_labels = {}  # string: int
        probs = {}  # int: float
        #phenotypes = {}  # int: array

        for i, gen in enumerate(self.genotypes):
            phenotype = mset.unique(gen)
            string = phenotype.tostring()
            if string not in phenotype_labels:
                label = i
                phenotype_labels[string] = label
                probs[label] = self.probabilities[i]
                #phenotypes[label] = phenotype
            else:
                label = phenotype_labels[string]
                probs[label] += self.probabilities[i]
            labels[i] = label

        phenotype_labels, probs = (zip(*probs.items()))
        mode = phenotype_labels[np.argmax(probs)]
        idx = labels == mode
        return self.genotypes[idx], self.probabilities[idx]


@dataclass
class GenotypeMultiTrace(object):
    """Generic class for multi-chain MCMC haplotype assembler trace.

    Attributes
    ----------
    genotypes : ndarray, int, shape (n_chains, n_steps, ploidy, n_positions)
        The genotype recorded at each step of a Markov chain Monte Carlo 
        simulation for a locus covering n_positions variable positions.
    llks : ndarray, float, shape (n_chains, n_steps)
        The log-likelihood calculated for the genotype at each step of
        the Markov chain Monte Carlo simulation.

    """

    genotypes: np.ndarray
    llks: np.ndarray

    def __post_init__(self):
        
        if (self.genotypes is not None) and (self.genotypes.shape[-1] !=0):

            self.genotypes=self.genotypes.copy()
            self.llks=self.llks.copy()
        
            assert np.ndim(self.genotypes) == 4
            assert np.ndim(self.llks) == 2
            assert self.genotypes.shape[0:2] == self.llks.shape

            # sort genotypes
            n_chains, n_steps = self.genotypes.shape[0:2]
            for c in range(n_chains):
                for i in range(n_steps):
                    self.genotypes[c, i] = allelic.sort(self.genotypes[c, i])


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
        # avoid calling __post_init__ because genotypes should 
        # already be sorted and sorting many genotypes is expensive
        new =  type(self)(
            None, 
            None,
        )
        new.genotypes = self.genotypes[:, n:]
        new.llks = self.llks[:, n:]
        return new

    def posterior(self):
        """Returns a posterior distribution over (phased) genotypes

        Returns
        -------
        posterior : PosteriorGenotypeDistribution
            A distribution over unique genotypes recorded in the trace.

        """
        # merge chains
        n_chain, n_step, ploidy, n_base = self.genotypes.shape
        genotypes = self.genotypes.reshape(n_chain * n_step, ploidy, n_base)

        states, counts = mset.unique_counts(genotypes)
        probs = counts / np.sum(counts)

        idx = np.flip(np.argsort(probs))

        return PosteriorGenotypeDistribution(states[idx], probs[idx])
    