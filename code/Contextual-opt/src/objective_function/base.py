#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import numpy as np
import sys, os
# sys.path.append(os.path.dirname(__file__) + "/../../")
from ..optimizer.base_optimizer import BaseOptimizer
from ..optimizer import cmaes as cma
from ..util import weight
from ..util import sampler as _sampler
from ..util import log

# public symbols
__all__ = ['ObjectiveFunction']


class ParametrizedObjectiveFunction(object):
    """
    Abstract Class for objective function
    :var int eval_count: evaluation count
    :var float best_eval: best evaluation value
    :var float target_eval: target evaluation value
    :var int max_eval: maximum number of evaluations
    :var bool minimization_problem: minimization problem or not
    """
    __metaclass__ = ABCMeta

    def __init__(self, max_eval, param=None, param_range = None):
        self.max_eval = max_eval
        self.best_eval = np.inf if self.minimization_problem else -np.inf
        self.best_solution = None

        self.eval_count = 0
        self.param = param
        self.param_range = param_range

        self.is_better = (lambda x, y: x < y) if self.minimization_problem else (lambda x, y: x > y)
        self.get_best = (lambda evals: np.min(evals)) if self.minimization_problem else (lambda evals: np.max(evals))
        self.get_best_idx = (lambda evals: np.argmin(evals)) if self.minimization_problem else (lambda evals: np.argmax(evals))

    def evaluation(self, X, param):
        if X.ndim == 1:
            X = X[None,:]
        
        Y = self._evaluation(X, param)
        if np.isscalar(Y):
            Y = np.array([Y])

        return Y

    @abstractmethod
    def _evaluation(self, X, param):
        """
        Abstract method for evaluation.
        :param X: candidate solutions
        :param: parameter
        :return: evaluation values
        """
        pass

    def __call__(self, X,param = []):
        """
        Return evaluation value of X.
        :param X: candidate solutions
        :return: evaluation values
        """
        self.eval_count += len(X)
        if len(param) == 0:
            evals = self.evaluation(X, self.param)
        else:
            evals = self.evaluation(X, param)
        self._update_best_eval(X, evals)
        return evals

    def clear(self):
        self.best_eval = np.inf if self.minimization_problem else -np.inf
        self.best_solution = None
        self.eval_count = 0

    def terminate_condition(self):
        """
        Check terminate condition.
        :return bool: terminate condition is satisfied or not
        """
        if self.eval_count >= self.max_eval:
            return True
        return False

    def verbose_display(self):
        """
        Return verbose display string.
        :return str: string for verbose display
        """
        return ' EvalCount: %d' % self.eval_count + ' BestEval: {}'.format(self.best_eval) + ' Param: {}'.format(self.param)

    @staticmethod
    def info_header():
        """
        Return string list of header.
        :return: string list of header
        """
        return ['EvalCount', 'BestEval', 'Param']

    def info_list(self):
        """
        Return string list of evaluation count and best evaluation value.
        :return: string list of evaluation count and best evaluation value
        """
        return ['%d' % self.eval_count, '%e' % self.best_eval, self.param]

    def _update_best_eval(self, X, evals):
        """
        Update best evaluation value.
        :param evals: new evaluation values
        :type evals: array_like, shape(lam), dtype=float
        """
        if self.is_better(self.get_best(evals), self.best_eval):
            self.best_eval = self.get_best(evals)
            self.best_solution = X[self.get_best_idx(evals)]
    
    def set_param(self, param):
        self.param = param

    def optimal_solution(self, param):
        """
        return optimal solution
        """
        return None
    
    def optimal_evaluation(self, param):
        """
        return evaluation value at optimal solution
        """
        return None




class ParameterFunction(object):
    """
    function of parameter variable
    Attributes:
        d_param: number of dimensions in parameter space
        d_lower: number of dimensions in design variable space
        objective_function: objective function with parameter variable
        optimizer : optimizer for optimizing the objective_function (optional)
        sampler: sampler for optimizer (optional)
    """
    def __init__(
        self,
        objective_function: ParametrizedObjectiveFunction,
        optimizer: BaseOptimizer = None,
        max_eval: int = 3e4,
        log_name: str = None,
        log_path: str = "./"
    ):
        self.objective_function = objective_function
        self.d_param = objective_function.param_dim
        self.d_lower = objective_function.d
        self.eval_count = 0
        self.max_eval = max_eval
        self.log_name = log_name
        self.log_path = log_path
        self.mean_eval = 0

        if optimizer is None:
            init_m, init_sigma = np.random.rand(self.d_lower) * 2 - 1, 2
            lam = cma.CMAParam.pop_size(self.d_lower)
            w_func = weight.CMAWeight(lam, min_problem=objective_function.minimization_problem)
            optimizer = cma._CMAES(self.d_lower, w_func, m=init_m, sigma=init_sigma)
            optimizer.set_terminate_condition()
        
        self.set_optimizer(optimizer)

    def __call__(
        self,
        param: np.ndarray,
        max_eval: int = None,
        return_mean: bool = False,
        noclear_fcall = False
    ):
        """
        return the best solution
        Arg:
            objective_function ObjectiveFunction:
        Return:
            SurrogateModel
        """
        if ~noclear_fcall:
            self.objective_function.clear()
            
        self.objective_function.set_param(param)
        self.objective_function.max_eval = self.max_eval - self.eval_count
        if self.max_eval is not None and self.max_eval < self.objective_function.max_eval:
            self.objective_function.max_eval = self.max_eval

        # ログとってる
        if self.log_name is not None :
            ex_log = log.DataLogger(self.log_name,self.log_path)
        else:
            ex_log = None

        self.optimizer.run(self.sampler, logger=ex_log, verbose=False)
        self.eval_count += self.objective_function.eval_count

        if return_mean:
            return self.optimizer.model.m
        else:
            return self.objective_function.best_solution
    
    def set_optimizer(self, optimizer, lam=None, sampler=None):
        self.optimizer = optimizer
        if lam is None:
            lam = cma.CMAParam.pop_size(self.d_lower)
        self.sampler = sampler if sampler is not None else _sampler.NoiseSampler(self.objective_function, lam) 