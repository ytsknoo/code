#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import numpy as np

# public symbols
__all__ = ['DefaultSampler', 'ImportanceMixingSampler']


class DefaultSampler(object):
    def __init__(self, f, lam):
        self.f = f
        self.lam = lam

    def __call__(self, model):
        X = model.sampling(self.lam)
        evals = self.f(X)
        return X, evals

    def verbose_display(self):
        return ''

    def log_header(self):
        return []

    def log(self):
        return []


class ImportanceMixingSampler(object):
    def __init__(self, f, lam, rate=0.01):
        self.f = f
        self.lam = lam
        self.rate = rate

        self.pX = None
        self.p_eval = None
        self.p_model = None
        self.accept_num = 0

    def __call__(self, model):
        if self.pX is not None:
            X = np.empty((self.lam, self.f.d))
            evals = np.empty(self.lam)

            # Rejection sampling from previous batch
            lll_pX_pM = self.p_model.loglikelihood(self.pX)
            lll_pX_cM = model.loglikelihood(self.pX)
            ll_ratio = np.exp(lll_pX_cM - lll_pX_pM)

            accept_idx = np.minimum(1., (1. - self.rate)*ll_ratio) > np.random.rand(self.lam)
            self.accept_num = accept_idx.sum()

            X[:self.accept_num] = self.pX[accept_idx]
            evals[:self.accept_num] = self.p_evals[accept_idx]

            # Reverse rejection sampling
            sample_num = self.accept_num
            while sample_num != self.lam:
                tmpX = model.sampling(self.lam - sample_num)
                lll_cX_pM = self.p_model.loglikelihood(tmpX)
                lll_cX_cM = model.loglikelihood(tmpX)
                ll_ratio = np.exp(lll_cX_pM - lll_cX_cM)

                accept_idx = np.maximum(self.rate, 1. - ll_ratio) > np.random.rand(self.lam - sample_num)
                n = accept_idx.sum()

                if n != 0:
                    X[sample_num:sample_num+n] = tmpX[accept_idx]
                    evals[sample_num:sample_num+n] = self.f(tmpX[accept_idx])
                    sample_num += n
        else:
            X = model.sampling(self.lam)
            evals = self.f(X)

        # previous solution information
        self.pX = X.copy()
        self.p_evals = evals.copy()
        self.p_model = copy.deepcopy(model)

        return X, evals

    def verbose_display(self):
        return ' Accept_num: %d' % self.accept_num

    def log_header(self):
        return ['accept_num']

    def log(self):
        return ['%d' % self.accept_num]


class BoxConstraintSampler(object):
    def __init__(self, f, lam, lower_bound, upper_bound):
        self.f = f
        self.lam = lam
        self.lower_bound = lower_bound 
        self.upper_bound = upper_bound

    def __call__(self, model):
        X = model.sampling(self.lam)
        X = np.minimum(np.maximum(X, self.lower_bound), self.upper_bound)
        evals = self.f(X)
        return X, evals

    def verbose_display(self):
        return ''

    def log_header(self):
        return []

    def log(self):
        return []


class NoiseSampler(DefaultSampler):
    def __init__(self, f, lam):
        self.f = f
        self.lam = lam
        self.n_eval = 1

    def __call__(self, model):
        X = model.sampling(self.lam)
        evals = np.zeros(len(X))

        for _ in range(self.n_eval):
            evals += self.f(X)
        evals /= self.n_eval

        return X, evals
    
    def reevaluate(self, X):
        evals = np.zeros(len(X))
        for _ in range(self.n_eval):
            evals += self.f(X)
        evals /= self.n_eval

        return evals
