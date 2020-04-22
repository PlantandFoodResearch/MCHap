import numpy as np
from dataclasses import dataclass
from haplohelper import mset
from haplohelper.encoding import allelic


@dataclass
class Assembler(object):
    """Abstract base class for haplotype assemblers.
    """
            
    @classmethod
    def parameterize(cls, *args, **kwargs):
        """Returns an instance with specified parameters.
        This is conveniance method to create a delayed instance of a class with Dask.
        """
        return cls(*args, **kwargs)


    def fit(self):
        """Fit an assembler model to a dataset.
        """
        raise NotImplementedError()


@dataclass
class PosteriorGenotypeDistribution(object):
    """Generic class for posterior distribution of (finite countable) genotypes.
    """

    genotypes: np.ndarray
    probabilities: np.ndarray

    def mode(self):
        idx = np.argmax(self.probabilities)
        return self.genotypes[idx], self.probabilities[idx]


@dataclass
class GenotypeTrace(object):
    """Generic class for single chained MCMC haplotype assembler trace.
    """

    genotypes: np.ndarray
    llks: np.ndarray

    def __post_init__(self):
        
        if self.genotypes is not None:

            self.genotypes=self.genotypes.copy()
            self.llks=self.llks.copy()
        
            assert np.ndim(self.genotypes) == 3
            assert np.ndim(self.llks) == 1
            assert len(self.genotypes) == len(self.llks)

            # sort genotypes
            for i in range(len(self.genotypes)):
                self.genotypes[i] = allelic.sort(self.genotypes[i])


    def burn(self, n):
        """Returns a Trace object without the first `n` observations.
        """
        # avoid calling __post_init__ because genotypes should already be sorted
        # and sorting many genotypes is expensive
        new =  type(self)(
            None, 
            None,
        )
        new.genotypes = self.genotypes[n:]
        new.llks = self.llks[n:]
        return new

    def posterior(self):
        """Returns a posterior distribution over (phased) genotypes
        """
        states, counts = mset.unique_counts(self.genotypes)
        probs = counts / np.sum(counts)

        idx = np.flip(np.argsort(probs))

        return PosteriorGenotypeDistribution(states[idx], probs[idx])

    



    