import numpy as np
import pymc3 as pm
from theano import tensor as tt
import pymc3
import theano
import biovector as bv


class BayesianHaplotypeModel(object):

    def trace_haplotypes(self, sort=True, **kwargs):
        raise NotImplementedError

    def trace_haplotype_counts(self, order='descending', **kwargs):
        trace = self.trace_haplotypes(**kwargs)
        n_steps, ploidy, n_base, n_nucl = trace.shape
        trace = trace.reshape((n_steps * ploidy, n_base, n_nucl))
        return bv.mset.counts(trace, order=order)

    def trace_haplotype_set_counts(self, order='descending', **kwargs):
        trace = self.trace_haplotypes(**kwargs)
        n_steps, ploidy, n_base, n_nucl = trace.shape
        trace = trace.reshape((n_steps, ploidy * n_base, n_nucl))
        sets, counts = bv.mset.counts(trace, order=order)
        return sets.reshape(-1, ploidy, n_base, n_nucl), counts

    def posterior(self, order='descending', **kwargs):
        sets, counts = self.trace_haplotype_set_counts(order=order, **kwargs)
        return sets, counts/np.sum(counts)


class BayesianHaplotypeAssembler(BayesianHaplotypeModel):

    def __init__(self, ploidy=None, prior=None, **kwargs):
        # check ploidy matches prior if given
        # check read shape matches prior if given
        self.ploidy = ploidy
        self.trace = None
        self.prior = prior
        self.fit_kwargs = kwargs

    def trace_haplotypes(self, sort=True, **kwargs):
        trace = self.trace.get_values('haplotypes', **kwargs)
        ploidy, n_base, n_nucl = trace.shape[-3:]
        trace = trace.reshape(-1,
                              ploidy,
                              n_base,
                              n_nucl).astype(np.int8)

        if sort:
            for i, haps in enumerate(trace):
                trace[i] = bv.mset.sort_onehot(haps)

        return trace

    def fit(self, reads):
        pass

        n_reads, n_base, n_nucl = reads.shape
        ploidy = self.ploidy

        if self.prior is None:
            null_nucleotide = np.ones(n_nucl) / n_nucl
            prior = np.array([null_nucleotide
                              for _
                              in range(n_base * ploidy)])
        else:
            prior = self.prior.reshape(n_base * ploidy, n_nucl)

        with pymc3.Model() as model:

            bases = pm.Categorical(name='bases',
                                   p=prior,
                                   shape=(n_base * ploidy))

            haplotypes = pm.Deterministic(
                'haplotypes',
                tt.extra_ops.to_one_hot(
                    bases,
                    nb_class=n_nucl
                ).reshape((1, ploidy, n_base, n_nucl))
            )

            # log(P) function, observations is a list of read array
            # in order: mum, dad, kid_0, kid_1, ...
            def logp(observations):
                # probabilities of reads drawn from inherited haplotypes
                step_0 = haplotypes * observations
                step_1 = tt.sum(step_0, axis=3)
                step_2 = tt.prod(step_1, axis=2)
                step_3 = tt.sum(step_2, axis=1)
                step_4 = tt.sum(tt.log(step_3), axis=0)

                return step_4

            # maximise log(P) for given observations
            pm.DensityDist('logp', logp, observed={
                'observations':
                reads.reshape(n_reads, 1, n_base, n_nucl)
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
        n_reads, n_base, n_nucl = reads.shape
        n_haps = self.reference_haplotypes.shape[0]

        ref_haplotypes = theano.shared(
            self.reference_haplotypes).reshape((1,
                                                n_haps,
                                                n_base,
                                                n_nucl))

        if self.prior is None:
            prior = np.repeat(1, n_haps) / n_haps
        else:
            prior = self.prior

        with pymc3.Model() as model:

            inheritance = pm.Categorical('inheritance',
                                         p=prior,
                                         shape=self.ploidy)
            inhrt = pm.Deterministic('inhrt', tt.sum(
                tt.extra_ops.to_one_hot(inheritance, nb_class=n_haps),
                axis=0).reshape((1, -1)))

            def logp(observations):
                step_0 = ref_haplotypes * observations
                step_1 = tt.sum(step_0, axis=3)
                step_2 = tt.prod(step_1, axis=2)
                step_inhrt = step_2 * inhrt
                step_3 = tt.sum(step_inhrt, axis=1)
                step_4 = tt.sum(tt.log(step_3), axis=0)

                return step_4

            # maximise log(P) for given observations
            pm.DensityDist('logp', logp, observed={
                'observations': reads.reshape(n_reads, 1, n_base, n_nucl)})

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

        parent_haps = theano.shared(np.concatenate(
            [self.maternal_haplotypes,
             self.paternal_haplotypes])).reshape(
            (1, n_mum_haps + n_dad_haps, n_base, n_nucl))

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

        # grid of priors with n_cats number of cells (combinations)
        n_cats = n_mum_haps * n_dad_haps
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
            mum_inhrt = tt.sum(
                tt.extra_ops.to_one_hot(maternal_inheritance,
                                        nb_class=n_mum_haps), axis=0)
            dad_inhrt = tt.sum(
                tt.extra_ops.to_one_hot(paternal_inheritance,
                                        nb_class=n_dad_haps), axis=0)
            inhrt = pm.Deterministic('inhrt', tt.concatenate(
                [mum_inhrt, dad_inhrt])).reshape((1, -1))

            def logp(observations):
                step_0 = parent_haps * observations
                step_1 = tt.sum(step_0, axis=3)
                step_2 = tt.prod(step_1, axis=2)
                step_inhrt = step_2 * inhrt
                step_3 = tt.sum(step_inhrt, axis=1)
                step_4 = tt.sum(tt.log(step_3), axis=0)

                return step_4

            # maximise log(P) for given observations
            pm.DensityDist('logp', logp, observed={
                'observations': reads.reshape(n_reads, 1, n_base, n_nucl)})

            # trace log likelihood
            llk = pm.Deterministic('llk', model.logpt)

            self.trace = pymc3.sample(**self.fit_kwargs)
