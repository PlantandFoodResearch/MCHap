import numpy as np
import pymc3 as pm
from theano import tensor as tt
import pymc3
import theano
import biovector as bv


def logp(reads, haplotypes):
    step_0 = reads * haplotypes
    step_1 = tt.sum(step_0, axis=3)
    step_2 = tt.prod(step_1, axis=2)
    step_3 = tt.sum(step_2, axis=1)
    step_4 = tt.sum(tt.log(step_3), axis=0)
    return step_4


class BayesianHaplotypeModel(object):

    def trace_haplotypes(self, sort=True, **kwargs):
        raise NotImplementedError

    def posterior_haplotypes(self, order='descending', counts=False, **kwargs):
        trace = self.trace_haplotypes(**kwargs)
        n_steps, ploidy, n_base, n_nucl = trace.shape
        trace = trace.reshape((n_steps * ploidy, n_base, n_nucl))
        haps, counts_ = bv.mset.count_unique(trace, order=order)
        if not counts:
            # then posterior probabilities
            counts_ = counts_/np.sum(counts_)
        return haps, counts_

    def posterior(self, order='descending', counts=False, **kwargs):
        trace = self.trace_haplotypes(**kwargs)
        n_steps, ploidy, n_base, n_nucl = trace.shape
        trace = trace.reshape((n_steps, ploidy * n_base, n_nucl))
        sets, counts_ = bv.mset.count_unique(trace, order=order)
        if not counts:
            # then posterior probabilities
            counts_ = counts_/np.sum(counts_)
        return sets.reshape(-1, ploidy, n_base, n_nucl), counts_


class BayesianHaplotypeAssembler(BayesianHaplotypeModel):

    def __init__(self, ploidy=None, prior=None, **kwargs):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.trace = None
        self.prior = prior
        self.fit_kwargs = kwargs
        # store read vector size when fitting to convert integers to one-hot
        self._vector_size = None

    def trace_integers(self, sort=True, **kwargs):
        trace = self.trace.get_values('integers', **kwargs)
        if sort:
            for i, haps in enumerate(trace):
                trace[i] = bv.integers.sort_integer_rows(haps)
        return trace

    def trace_haplotypes(self, sort=True, **kwargs):
        integers = self.trace_integers(sort=sort, **kwargs)
        return bv.integers.integers_as_onehot(integers, self._vector_size)

    def fit(self, reads):
        n_reads, n_base, n_nucl = reads.shape
        ploidy = self.ploidy

        # store read vector size for conversion of integers to one-hot
        # it's more memory efficient to only store the integer encoding
        self._vector_size = n_nucl

        if self.prior is None:
            prior = np.ones(n_nucl) / n_nucl
        else:
            prior = self.prior

        with pymc3.Model() as model:

            # propose haplotypes from categorical distribution
            # these are stored in integer encoding in the trace
            integers = pm.Categorical(
                name='integers',
                p=prior,
                shape=(ploidy, n_base)
            )

            # convert to one-hot encoding (floats) for log(P) calculation
            haplotypes = tt.extra_ops.to_one_hot(
                integers.reshape((ploidy * n_base,)),
                nb_class=n_nucl
            ).reshape((1, ploidy, n_base, n_nucl))

            # maximise log(P) for given observations
            pm.DensityDist('logp', logp, observed={
                'reads': reads.reshape(n_reads, 1, n_base, n_nucl),
                'haplotypes': haplotypes
            })

            # trace log likelihood
            llk = pm.Deterministic('llk', model.logpt)

            trace = pymc3.sample(**self.fit_kwargs)

        self.trace = trace


class BayesianDosageCaller(BayesianHaplotypeModel):

    def __init__(self,
                 ploidy=None,
                 reference_haplotypes=None,
                 prior=None,
                 **kwargs):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.reference_haplotypes = reference_haplotypes
        self.trace = None
        self.prior = prior
        self.fit_kwargs = kwargs

    def trace_inheritance(self, **kwargs):
        return np.sort(self.trace.get_values('inheritance',
                                             **kwargs),
                       axis=1)

    def posterior_inheritance(self, **kwargs):
        trace = self.trace_inheritance(**kwargs)
        array = np.zeros(len(self.reference_haplotypes))
        for i in trace.ravel():
            array[i] += 1
        return array / len(trace)

    def trace_haplotypes(self, sort=True, **kwargs):
        n_base, n_nucl = self.reference_haplotypes.shape[-2:]
        calls = self.trace_inheritance(**kwargs)
        trace = np.zeros((len(calls),
                          self.ploidy,
                          n_base,
                          n_nucl),
                         dtype=np.int8)

        for i, idx in enumerate(calls):
            trace[i] = self.reference_haplotypes[idx]
            if sort:
                trace[i] = bv.mset.sort_onehot(trace[i])

        return trace

    def fit(self, reads):
        ploidy = self.ploidy
        n_reads, n_base, n_nucl = reads.shape
        n_haps = self.reference_haplotypes.shape[0]

        ref_haplotypes = theano.shared(
            self.reference_haplotypes)

        if self.prior is None:
            prior = np.repeat(1, n_haps) / n_haps
        else:
            prior = self.prior

        with pymc3.Model() as model:

            # indices of reference haplotypes to select
            inheritance = pm.Categorical('inheritance',
                                         p=prior,
                                         shape=self.ploidy)

            # select haplotypes from reference set
            haplotypes = ref_haplotypes[inheritance]

            # maximise log(P) for given observations
            pm.DensityDist('logp', logp, observed={
                'reads': reads.reshape(n_reads, 1, n_base, n_nucl),
                'haplotypes': haplotypes.reshape((1, ploidy, n_base, n_nucl))
            })

            # trace log likelihood
            llk = pm.Deterministic('llk', model.logpt)

            self.trace = pymc3.sample(**self.fit_kwargs)


class BayesianChildDosageCaller(BayesianHaplotypeModel):

    def __init__(self,
                 ploidy=None,
                 maternal_haplotypes=None,
                 paternal_haplotypes=None,
                 maternal_prior=None,
                 paternal_prior=None,
                 **kwargs):
        self.ploidy = ploidy
        self.maternal_haplotypes = maternal_haplotypes
        self.paternal_haplotypes = paternal_haplotypes
        self.trace = None
        self.maternal_prior = maternal_prior
        self.paternal_prior = paternal_prior
        self.fit_kwargs = kwargs

    def trace_maternal_inheritance(self, **kwargs):
        return np.sort(self.trace.get_values('maternal_inheritance',
                                             **kwargs),
                       axis=1)

    def trace_paternal_inheritance(self, **kwargs):
        return np.sort(self.trace.get_values('paternal_inheritance',
                                             **kwargs),
                       axis=1)

    def posterior_maternal_inheritance(self, **kwargs):
        trace = self.trace_maternal_inheritance(**kwargs)
        array = np.zeros(len(self.maternal_haplotypes))
        for i in trace.ravel():
            array[i] += 1
        return array / len(trace)

    def posterior_paternal_inheritance(self, **kwargs):
        trace = self.trace_paternal_inheritance(**kwargs)
        array = np.zeros(len(self.paternal_haplotypes))
        for i in trace.ravel():
            array[i] += 1
        return array / len(trace)

    def trace_haplotypes(self, sort=True, **kwargs):
        n_base, n_nucl = self.maternal_haplotypes.shape[-2:]
        mum_trace = self.trace_maternal_inheritance(**kwargs)
        dad_trace = self.trace_paternal_inheritance(**kwargs)

        trace = np.zeros((len(mum_trace),
                          self.ploidy,
                          n_base,
                          n_nucl),
                         dtype=np.int8)

        for i, (mum_idx, dad_idx) in enumerate(zip(mum_trace, dad_trace)):
            trace[i] = np.concatenate([self.maternal_haplotypes[mum_idx],
                                       self.paternal_haplotypes[dad_idx]])
            if sort:
                trace[i] = bv.mset.sort_onehot(trace[i])

        return trace

    def fit(self, reads):
        n_reads, n_base, n_nucl = reads.shape
        ploidy = self.ploidy
        n_mum_haps = len(self.maternal_haplotypes)
        n_dad_haps = len(self.paternal_haplotypes)

        maternal_haplotypes = theano.shared(self.maternal_haplotypes)
        paternal_haplotypes = theano.shared(self.paternal_haplotypes)

        if self.maternal_prior is None:
            # flat prior
            maternal_prior = np.repeat(1, n_mum_haps) / n_mum_haps
        else:
            maternal_prior = self.maternal_prior

        if self.paternal_prior is None:
            # flat prior
            paternal_prior = np.repeat(1, n_dad_haps) / n_dad_haps
        else:
            paternal_prior = self.paternal_prior

        # grid of priors
        prior = np.ones((n_mum_haps, n_dad_haps), dtype=np.float)
        prior = maternal_prior.reshape(-1, 1) * prior
        prior = paternal_prior.reshape(1, -1) * prior

        with pymc3.Model() as model:
            inheritance = pm.Categorical(
                'inheritance',
                p=prior.ravel(),
                shape=ploidy//2
            )

            maternal_inheritance = pm.Deterministic(
                'maternal_inheritance',
                inheritance // n_dad_haps
            )
            paternal_inheritance = pm.Deterministic(
                'paternal_inheritance',
                inheritance % n_dad_haps
            )

            # select haplotypes from reference sets
            mum_haps = maternal_haplotypes[maternal_inheritance]
            dad_haps = paternal_haplotypes[paternal_inheritance]
            haplotypes = tt.concatenate([mum_haps, dad_haps])

            # maximise log(P) for given observations
            pm.DensityDist('logp', logp, observed={
                'reads': reads.reshape(n_reads, 1, n_base, n_nucl),
                'haplotypes': haplotypes.reshape((1, ploidy, n_base, n_nucl))
            })

            # trace log likelihood
            llk = pm.Deterministic('llk', model.logpt)

            self.trace = pymc3.sample(**self.fit_kwargs)
