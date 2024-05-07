#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg

from ..util.weight import CMAWeight
from ..optimizer.base_optimizer import BaseOptimizer
from ..util.model import GaussianSigmaC

# public symbols
__all__ = ['CMAParam', 'CMAES', '_CMAES']


class CMAParam(object):
    """
    Default parameters for CMA-ES.
    """
    @staticmethod
    def pop_size(dim):
        return 4 + int(np.floor(3 * np.log(dim)))

    @staticmethod
    def mu_eff(lam, weights=None):
        if weights is None and lam < 4:
            weights = CMAWeight(4).w
        if weights is None:
            weights = CMAWeight(lam).w
        w_1 = np.absolute(weights).sum()
        return w_1**2 / weights.dot(weights)

    @staticmethod
    def c_1(dim, mueff):
        return 2.0 / ((dim + 1.3) * (dim + 1.3) + mueff)

    @staticmethod
    def c_mu(dim, mueff, c1=0., alpha_mu=2.):
        return np.minimum(1. - c1, alpha_mu * (mueff - 2. + 1./mueff) / ((dim + 2.)**2 + alpha_mu * mueff / 2.))

    @staticmethod
    def c_c(dim, mueff):
        return (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)

    @staticmethod
    def c_sigma(dim, mueff):
        return (mueff + 2.0) / (dim + mueff + 5.0)

    @staticmethod
    def damping(dim, mueff):
        return 1.0 + 2.0 * np.maximum(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + CMAParam.c_sigma(dim, mueff)

    @staticmethod
    def chi_d(dim):
        return np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim**2))  # ||N(0,I)||


class CMAES(BaseOptimizer):
    def __init__(self, d, weight_func, m=None, C=None, sigma=1., minimal_eigenval=1e-30,
                 lam=None, c_m=1., c_1=None, c_mu=None, c_c=None, c_sigma=None, damping=None, alpha_mu=2.,sampling_mean=False):
        self.model = GaussianSigmaC(d, m=m, C=C, sigma=sigma, minimal_eigenval=minimal_eigenval,sampling_mean=sampling_mean)
        self.weight_func = weight_func
        self.lam = lam if lam is not None else CMAParam.pop_size(d)

        # CMA parameters
        self.mu_eff = CMAParam.mu_eff(self.lam)
        self.c_1 = CMAParam.c_1(d, self.mu_eff) if c_1 is None else c_1
        self.c_mu = CMAParam.c_mu(d, self.mu_eff, c1=self.c_1, alpha_mu=alpha_mu) if c_mu is None else c_mu
        self.c_c = CMAParam.c_c(d, self.mu_eff) if c_c is None else c_c
        self.c_sigma = CMAParam.c_sigma(d, self.mu_eff) if c_sigma is None else c_sigma
        self.damping = CMAParam.damping(d, self.mu_eff) if damping is None else damping
        self.chi_d = CMAParam.chi_d(d)
        self.c_m = c_m

        # evolution path
        self.ps = np.zeros(d)
        self.pc = np.zeros(d)
        self.gen_count = 0

        self.best_eval = np.inf

    def sampling_model(self):
        return self.model

    def update(self, X, evals):
        self.gen_count += 1

        # natural gradient
        weights = self.weight_func(evals)
        Y = (X - self.model.m) / self.model.sigma
        WYT = weights * Y.T
        m_grad = self.model.sigma * WYT.sum(axis=1)
        C_grad = np.dot(WYT, Y) - weights.sum() * self.model.C

        hsig = 1.
        if self.c_1 != 0. or self.damping != np.inf:
            # evolution path
            self.ps = (1.0 - self.c_sigma) * self.ps + np.sqrt(self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff) * np.dot(self.model.invSqrtC, self.c_m * m_grad / self.model.sigma)
            hsig = 1. if scipy.linalg.norm(self.ps) / (np.sqrt(1. - (1. - self.c_sigma) ** (2 * self.gen_count))) < (1.4 + 2. / (self.model.d + 1.)) * self.chi_d else 0.
            self.pc = (1. - self.c_c) * self.pc + hsig * np.sqrt(self.c_c * (2. - self.c_c) * self.mu_eff) * self.c_m * m_grad / self.model.sigma

        if self.damping != np.inf:
            # CSA
            self.model.sigma = self.model.sigma * np.exp(self.c_sigma / self.damping * (scipy.linalg.norm(self.ps) / self.chi_d - 1.))
        # mean vector update
        self.model.m = self.model.m + self.c_m * m_grad
        # covariance matrix update
        self.model.C = self.model.C + (1.-hsig)*self.c_1*self.c_c*(2.-self.c_c)*self.model.C + self.c_1 * (np.outer(self.pc, self.pc) - self.model.C) + self.c_mu * C_grad

    def terminate_condition(self):
        return self.model.terminate_condition()

    def verbose_display(self):
        return self.model.verbose_display()

    # def log_header(self):
    #     return self.model.log_header()

    # def log(self):
    #     return self.model.log()
    
       #log用に追加
    def log_header(self):
        return ['MeanEval'] + self.model.log_header()

    def log(self):
        mean_eval = self.objective_func.evaluation(self.model.m,self.objective_func.param)
        return ['%e' % mean_eval] + self.model.log()



class _CMAES(CMAES):
    """
    CMAES with terminate conditions
    """
    def __init__(self, d, weight_func, cov_eigval_target=1e-2, sampling_mean=False,fin_cov=False,**kwargs):
        super(_CMAES, self).__init__(d, weight_func, **kwargs,sampling_mean=sampling_mean)
        
        len_hist = np.max((100 / self.lam, 1)).astype(int)
        self.best_evals_hist = np.ones(len_hist)
        self.set_terminate_condition(cov_eigval_target)
        self.max_ite = np.inf
        self.fin_cov = fin_cov

    def set_terminate_condition(self, cov_eigval_target=1e-2, max_ite=np.inf):
        self.cov_eigval_target = cov_eigval_target
        self.max_ite = max_ite

    def terminate_condition(self):
        if self.fin_cov:
            # model_cov_eigvals = (self.model.sigma ** 2) * self.model.eigvals
            # if max(model_cov_eigvals) < self.cov_eigval_target:
            #     return True
            if self.eval_cons_flag > 50:
                return True
            
        if min(self.best_evals_hist) < 1e-8:
            # print("done1")
            return True
        # if self.gen_count > len(self.best_evals_hist) and max(self.best_evals_hist) - min(self.best_evals_hist) < 1e-5:
        #     print("done2")
        #     return True
        if self.gen_count > self.max_ite:
            # print("done2")
            return True
        # return self.model.terminate_condition()
    
    def update(self, X, evals):
        super().update(X, evals)

        self.best_evals_hist[self.gen_count % len(self.best_evals_hist)] = np.min(evals)
    
    def set_model_param(self, mean=None, std=None, cov=None):
        if mean is not None:
            self.model.m = mean
        if std is not None:
            self.model.sigma = std
        if cov is not None:
            self.model.C = cov
        return
    
    def set_evolution_path(self, pc=None, ps=None):
        if pc is not None:
            self.pc = pc
        if ps is not None:
            self.ps = ps
        return
    
    def get_evolution_path(self):
        return [self.pc, self.ps]


class _BoxConstraintCMAES(_CMAES):
    """
    CMAES with terminate conditions
    """
    def __init__(self, d, weight_func, constraint_upper, constraint_lower, **kwargs):
        super(_BoxConstraintCMAES, self).__init__(d, weight_func, **kwargs)
        self.constraint_upper = constraint_upper
        self.constraint_lower = constraint_lower

        self.gamma_constraint = np.zeros(d)
        self.is_set_boundary_weight = False
        self.len_hist_iqr = int(20 + np.floor(3 * d / self.lam))
        self.hist_iqr = np.zeros(self.len_hist_iqr)

        self.pycma_update = True
    
    def update(self, X, evals):
        # --
        # this implmentation is based on pycma (some modification is included compared to [Hansen 2009])
        # see https://github.com/CMA-ES/pycma/blob/ac4a913b4f2c0ff61c18ee29695bbe732a3828d3/cma/constraints_handler.py#L418
        # [Hansen 2009]: Hansen et al 2009, A Method for Handling Uncertainty... IEEE TEC
        # --

        evals_q75, evals_q25 = np.percentile(evals, [75 ,25])
        if self.pycma_update: ### latest version
            self.hist_iqr[self.gen_count % self.len_hist_iqr] = \
                (evals_q75 - evals_q25) / (self.model.sigma ** 2) / np.diag(self.model.C).mean()
        else: ### original version
            self.hist_iqr[self.gen_count % self.len_hist_iqr] = evals_q75 - evals_q25


        # check if the mean vector is out-of-bound
        mean_infeasible = np.max(np.stack((
                self.model.m - self.constraint_upper, 
                self.constraint_lower - self.model.m, 
                np.zeros_like(self.model.m))), axis=0)
        mean_dist_infeasible = np.abs(mean_infeasible)
        is_mean_out = np.sum(mean_dist_infeasible) > 0

        # set boundary weight
        max_hist = np.min((self.gen_count+1, self.len_hist_iqr)).astype(int)
        delta_fit = np.median(self.hist_iqr[:max_hist])
        if is_mean_out and (not self.is_set_boundary_weight or self.gen_count == 1):
            self.is_set_boundary_weight = True
            if self.pycma_update: ### latest version
                self.gamma_constraint = 2 * delta_fit
            else: ### original version
                self.gamma_constraint = 2 * delta_fit / (self.model.sigma ** 2) / np.diag(self.model.C).mean()
            
        # update boundary_weight
        if self.is_set_boundary_weight:
            coordinate_threshold = \
                    3 * self.model.sigma * np.sqrt(np.diag(self.model.C)) * \
                    np.max((1, np.sqrt(self.model.d) / self.mu_eff))
            if self.pycma_update: ### latest version
                mean_dist_infeasible_trans = \
                    (mean_dist_infeasible - coordinate_threshold) \
                    / (self.model.sigma * np.sqrt(np.diag(self.model.C)))
                mean_dist_infeasible_trans = np.maximum(mean_dist_infeasible_trans, 0)
                damp = np.min((1, self.mu_eff / (10 * self.model.d)))
                self.gamma_constraint *= \
                    np.exp(np.tanh((mean_dist_infeasible_trans) / 3) * damp / 2)
                self.gamma_constraint[self.gamma_constraint > 5 * delta_fit] *=  np.exp(- damp / 3)
            else: ### original version
                is_mean_out_strict = mean_dist_infeasible > coordinate_threshold
                self.gamma_constraint *= 1.1 ** (is_mean_out_strict * np.max((1, self.mu_eff / (10 * self.model.d))))


        # check element-wise distance from the boundary (if out-of-bound)
        dist_infeasible = np.max(np.stack((
                X - self.constraint_upper, 
                self.constraint_lower - X, 
                np.zeros_like(X))), axis=0)
        
        if self.pycma_update: ### latest version
            penalty = ((dist_infeasible ** 2) * self.gamma_constraint).mean(axis=1)
        else: ### original version
            xi_constraint = np.exp(0.9 * (np.log(np.diag(self.model.C)) - np.log(np.diag(self.model.C).mean())))
            penalty = ((dist_infeasible ** 2) * self.gamma_constraint / xi_constraint).mean(axis=1)
        
        super(_BoxConstraintCMAES, self).update(X, evals + penalty)



class _BoxConstraintUHCMAES(_BoxConstraintCMAES):
    def __init__(self, d, weight_func, constraint_upper, constraint_lower, **kwargs):
        super(_BoxConstraintUHCMAES, self).__init__(d, weight_func, constraint_upper, constraint_lower, **kwargs)
        
        self.uh_r_lam = 0.3
        self.uh_theta = 0.2
        self.uh_alpha = 1.5
        self.n_eval = 1

        self.objective_func = None
        

    def run(self, sampler, logger=None, verbose=True):
        """
        Running script of Information Geometric Optimization (IGO)
        """

        f = sampler.f

        if logger is not None:
            logger.write_csv(['Ite'] + f.info_header() + self.log_header() + sampler.log_header() )

        ite = 0

        while not sampler.f.terminate_condition() and not self.terminate_condition():
            ite += 1

            # sampling and evaluation
            sampler.n_eval = np.max((int(self.n_eval + 0.5), 1))

            X, evals = sampler(self.sampling_model())

            ceil_reeval_ram = int(self.uh_r_lam * len(evals))
            add_reeval_lam = np.random.rand() < (self.uh_r_lam * len(evals) - ceil_reeval_ram)
            reeval_lam = np.max((ceil_reeval_ram + add_reeval_lam, 1))
            reevals = sampler.reevaluate(X[:reeval_lam])

            # display and save log
            if verbose:
                print(str(ite) + f.verbose_display() + self.verbose_display() + sampler.verbose_display())
            if logger is not None:
                logger.write_csv([str(ite)] + f.info_list() + self.log() + sampler.log())

            # parameter update
            self.update(X, evals, reevals)
            
            # update_sampler
            # TODO

        return [f.eval_count, f.best_eval]
    
    def update_sampler(self):
        return

    def update(self, X, evals, reevals):

        evals_new = evals
        evals_new[:len(reevals)] = reevals

        evals_all = np.r_[evals, evals_new]

        ranking_all = np.argsort(np.argsort(evals_all))
        ranking_old = ranking_all[:len(reevals)]
        ranking_new = ranking_all[len(evals):len(evals)+len(reevals)]
        delta = ranking_new - ranking_old - (np.sign(np.abs(ranking_old - ranking_new)))

        def delta_theta(R):
            positive_set = np.abs(np.arange(1, 2 * len(evals_all) - 1)[:, np.newaxis] - R[np.newaxis, :])
            return np.quantile(positive_set, self.uh_theta * 0.5, axis=0)

        R_new = ranking_new - (reevals > evals[:len(reevals)])
        R_old = ranking_old - (reevals < evals[:len(reevals)])
        
        s = (2 * np.abs(delta) - delta_theta(R_new) - delta_theta(R_old)).mean()

        if s > 0:
            self.n_eval *= self.uh_alpha
        else:
            self.n_eval /= self.uh_alpha
        
        super().update(X, (evals + evals_new) / 2.)
    
    def set_n_eval(self, n_eval):
        self.n_eval = n_eval
        return