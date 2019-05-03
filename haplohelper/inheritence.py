#!/usr/bin/env python3

import numpy as np
from biovector import mset


class TrioChildInheritance(object):

    def __init__(self, maternal_haplotypes, paternal_haplotypes):

        self.mum = maternal_haplotypes.copy()
        self.dad = paternal_haplotypes.copy()

        assert self.mum.ndim == self.dad.ndim == 3
        assert self.mum.shape == self.dad.shape
        assert self.mum.dtype == self.dad.dtype

        self.ploidy, self.n_base, self.n_nucl = self.mum.shape
        self.dtype = self.mum.dtype

        self.reset()

    def reset(self):
        self.taken = np.zeros((0, self.n_base, self.n_nucl), self.dtype)
        self.mum_taken = self.taken.copy()
        self.dad_taken = self.taken.copy()

    def available(self):
        result = np.zeros((0, self.n_base, self.n_nucl), self.dtype)
        if len(self.taken) == self.ploidy:
            return result
        if len(self.mum_taken) < (self.ploidy / 2):
            result = mset.add(result, mset.subtract(self.mum, self.mum_taken))
        if len(self.dad_taken) < (self.ploidy / 2):
            result = mset.add(result, mset.subtract(self.dad, self.dad_taken))
        return mset.unique(result)

    def take(self, hap, check=True):

        if hap.ndim == 2:
            hap = np.expand_dims(hap, 0)

        if check:
            assert mset.contains(self.available(), hap)

        self.taken = mset.add(self.taken, hap)

        improving = True
        while improving:
            improving = False

            # taken
            un_assigned = mset.subtract(self.taken,
                                        mset.add(self.mum_taken,
                                                 self.dad_taken))

            if len(self.dad_taken) == self.ploidy / 2:
                mum_update = mset.intercept(self.mum, un_assigned)
            else:
                dad_slots = (self.ploidy / 2) - len(self.dad_taken)
                dad_heritable = mset.thin(mset.subtract(self.dad,
                                                        self.dad_taken),
                                          dad_slots)
                mum_update = mset.intercept(self.mum,
                                            mset.subtract(un_assigned,
                                                          dad_heritable))

            if len(mum_update) > 0:
                self.mum_taken = mset.add(self.mum_taken, mum_update)
                improving = True

            un_assigned = mset.subtract(self.taken,
                                        mset.add(self.mum_taken,
                                                 self.dad_taken))

            if len(self.mum_taken) == self.ploidy / 2:
                dad_update = mset.intercept(self.dad, un_assigned)
            else:
                mum_slots = (self.ploidy / 2) - len(self.mum_taken)
                mum_heritable = mset.thin(mset.subtract(self.mum,
                                                        self.mum_taken),
                                          mum_slots)
                dad_update = mset.intercept(self.dad,
                                            mset.subtract(un_assigned,
                                                          mum_heritable))

            if len(dad_update) > 0:
                self.dad_taken = mset.add(self.dad_taken, dad_update)
                improving = True
