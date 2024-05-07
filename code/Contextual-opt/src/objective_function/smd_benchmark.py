#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib.pyplot import axis
from .base import ParametrizedObjectiveFunction
import numpy as np
from abc import abstractmethod

# public symbols
__all__ = []


class SMD(object):
    minimization_problem = True

    def __init__(self, d_upper, d_lower, d_common=None, max_lower_eval=np.inf, max_upper_eval=np.inf):
        self.d_upper = d_upper
        self.d_lower = d_lower
        self.d_common = d_common if d_common is not  None else int(np.floor(d_upper / 2))

        assert self.d_common < np.min((self.d_upper, self.d_lower)), "SMD: both of total num. of dim. for lower and upper should be strictly larger than d_common."

        self.upper_eval_count = 0
        self.lower_eval_count = 0

        self.max_upper_eval = max_upper_eval
        self.max_lower_eval = max_lower_eval

        self.set_constraint_num()

    def set_constraint_num(self):
        self.upper_constraint_num = 0 
        self.lower_constraint_num = 0 
        
    def evaluation_upper(self, X_upper, X_lower):
        self.upper_eval_count += len(X_upper)
        return self._evaluation_upper(X_upper, X_lower)
    
    def evaluation_lower(self, X_upper, X_lower):
        self.lower_eval_count += len(X_upper)
        return self._evaluation_lower(X_upper, X_lower)

    def _evaluation_upper(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        return self._F1(X_u1) + self._F2(X_l1) + self._F3(X_u2, X_l2)
        
    def _evaluation_lower(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        return self._f1(X_u1, X_u2) + self._f2(X_l1) + self._f3(X_u2, X_l2)
    
    @abstractmethod
    def upper_domain(self, d):
        pass

    @abstractmethod
    def lower_domain(self, d):
        pass
    
    @abstractmethod
    def upper_optimal_solution(self):
        pass

    @abstractmethod
    def lower_optimal_solution(self, X_upper):
        pass

    @abstractmethod
    def _F1(self, X_u1):
        pass

    @abstractmethod
    def _F2(self, X_l1):
        pass

    @abstractmethod
    def _F3(self, X_u2, X_l2):
        pass

    @abstractmethod
    def _f1(self, X_u1, X_u2):
        pass

    @abstractmethod
    def _f2(self, X_l1):
        pass

    @abstractmethod
    def _f3(self, X_u2, X_l2):
        pass

    def upper_constraint(self, X_upper, X_lower):
        sample_num = X_upper.shape[0]
        return np.zeros(sample_num)

    def lower_constraint(self, X_upper, X_lower):
        sample_num = X_upper.shape[0]
        return np.zeros(sample_num)

    def is_lower_optimal_solution(self, X_upper, X_lower):
        return np.abs(self.lower_optimal_solution(X_upper) - X_lower).sum(axis=1) == 0

    def lower_optimal_solution_with_upper_opt(self):
        return self.lower_optimal_solution(self.upper_optimal_solution()[None,:])[0]


class SMD1(SMD):
    def upper_domain(self):
        upper_domain_min = - 5 * np.ones(self.d_upper)
        upper_domain_max = 10 * np.ones(self.d_upper)
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), - np.pi / 2 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.pi / 2 * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), np.arctan(X_upper[:, -self.d_common:])]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) + ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)


class SMD2(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = - 5 * np.ones(self.d_upper)
        upper_domain_max = np.r_[10 * np.ones(d_diff), 1 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), 1e-10 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.e * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), np.exp(X_upper[:, -self.d_common:])]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return - (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1)


class SMD3(SMD):
    def upper_domain(self):
        upper_domain_min = - 5 * np.ones(self.d_upper)
        upper_domain_max = 10 * np.ones(self.d_upper)
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), - np.pi / 2 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.pi / 2 * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), np.arctan(X_upper[:, -self.d_common:] ** 2)]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) + ((X_u2**2 - np.tan(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        d_diff = self.d_lower - self.d_common
        return d_diff + (X_l1**2 - np.cos(2 * np.pi * X_l1)).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2**2 - np.tan(X_l2)) ** 2).sum(axis=1)


class SMD4(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[- 5 * np.ones(d_diff), - 1 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 1 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), 0 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.e * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), np.exp(np.abs(X_upper[:, -self.d_common:])) - 1]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return - (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((np.abs(X_u2) - np.log(X_l2 + 1)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        d_diff = self.d_lower - self.d_common
        return d_diff + (X_l1**2 - np.cos(2 * np.pi * X_l1)).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((np.abs(X_u2) - np.log(X_l2 + 1)) ** 2).sum(axis=1)


class SMD5(SMD):
    def upper_domain(self):
        upper_domain_min = - 5 * np.ones(self.d_upper)
        upper_domain_max = 10 * np.ones(self.d_upper)
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        lower_domain_min = - 5 * np.ones(self.d_lower)
        lower_domain_max = 10 * np.ones(self.d_lower)
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.ones((X_upper.shape[0], d_diff)), np.sqrt(np.abs(X_upper[:, -self.d_common:]))]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return - ((X_l1[:, 1:] - X_l1[:, :-1]**2).sum(axis=1) + ((X_l1 - 1)**2).sum(axis=1))

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((np.abs(X_u2) - X_l2**2) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1[:, 1:] - X_l1[:, :-1]**2).sum(axis=1) + ((X_l1 - 1)**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((np.abs(X_u2) - X_l2 **2) ** 2).sum(axis=1)


class SMD6(SMD):
    s = 2 # probably should be even number

    def upper_domain(self):
        upper_domain_min = - 5 * np.ones(self.d_upper)
        upper_domain_max = 10 * np.ones(self.d_upper)
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        lower_domain_min = - 5 * np.ones(self.d_lower)
        lower_domain_max = 10 * np.ones(self.d_lower)
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), X_upper[:, -self.d_common:]]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        d_diff = self.d_lower - self.d_common
        assert d_diff - self.s >= 0, "SMD6: d_lower - d_common should be larger than s={}".format(self.s)
        return - (X_l1[:, :-self.s]**2).sum(axis=1) + (X_l1[:, -self.s:]**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((X_u2 - X_l2) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        d_diff = self.d_lower - self.d_common
        assert d_diff - self.s >= 0, "SMD6: d_lower - d_common should be larger than s={}".format(self.s)
        return (X_l1[:, :-self.s]**2).sum(axis=1) + ((X_l1[:, (- self.s + 1)::2] - X_l1[:, -self.s:-1:2])**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - X_l2) ** 2).sum(axis=1)


class SMD7(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[- 5 * np.ones(d_diff), - 5 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 1 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), 0 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.e * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), np.exp(X_upper[:, -self.d_common:])]

    def _F1(self, X_u1):
        d_diff = self.d_upper - self.d_common
        return 1 + (X_u1**2).sum(axis=1) / 400 - ( np.cos(X_u1 / np.sqrt(np.arange(1, d_diff+1))) ).prod(axis=1)

    def _F2(self, X_l1):
        return - (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**3).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1)


class SMD8(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[- 5 * np.ones(d_diff), - 5 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 10 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), -5 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), 10 * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        # this differs from (35) in https://www.egr.msu.edu/~kdeb/papers/k2013002.pdf
        # refer an implementation in https://github.com/jmejia8/bilevel-benchmark/blob/05437a48a1ee0fd5c2c464e8affc6994b4136582/smd.c#L145
        return np.c_[np.ones((X_upper.shape[0], d_diff)), X_upper[:, -self.d_common:] ** (1./3)]

    def _F1(self, X_u1):
        d_diff = self.d_upper - self.d_common
        return 20 + np.e - 20 * np.exp(-0.2 * np.sqrt((X_u1**2).sum(axis=1) / d_diff)) - np.exp(np.cos(2 * np.pi * X_u1).sum(axis=1) / d_diff)

    def _F2(self, X_l1):
        return - ((X_l1[:, 1:] - X_l1[:, :-1]**2).sum(axis=1) + ((X_l1 - 1)**2).sum(axis=1))

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((X_u2 - (X_l2 ** 3)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return np.abs(X_u1).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1[:, 1:] - X_l1[:, :-1]**2).sum(axis=1) + ((X_l1 - 1)**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - (X_l2 ** 3)) ** 2).sum(axis=1)


class SMD9(SMD):
    constraint_coef_a = 1
    constraint_coef_b = 1

    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[- 5 * np.ones(d_diff), - 5 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 1 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), -1 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), (-1+np.e) * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        return np.c_[np.zeros((X_upper.shape[0], d_diff)), np.exp(np.abs(X_upper[:, -self.d_common:])) - 1]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return - (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((X_u2 - np.log(X_l2 + 1)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.log(X_l2 + 1)) ** 2).sum(axis=1)
    
    def set_constraint_num(self):
        self.upper_constraint_num = 1
        self.lower_constraint_num = 1
        
    def upper_constraint(self, X_upper, X_lower):
        #X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        constraint_1 = ((X_u1**2).sum(axis=1) + (X_u2**2).sum(axis=1)) / self.constraint_coef_a
        constraint_2 = constraint_1 + 0.5 / self.constraint_coef_b
        return np.array([constraint_1 - np.floor(constraint_2)])

    def lower_constraint(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        #X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        constraint_1 = ((X_l1**2).sum(axis=1) + (X_l2**2).sum(axis=1)) / self.constraint_coef_a
        constraint_2 = constraint_1 + 0.5 / self.constraint_coef_b
        return np.array([constraint_1 - np.floor(constraint_2)])


class SMD10(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[-5 * np.ones(d_diff), - 5 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 10 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        eps = 1e-8
        lower_domain_min = np.r_[-5 * np.ones(d_diff), -np.pi/2 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.pi/2 * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.ones(self.d_upper) / np.sqrt(self.d_upper-1)

    def lower_optimal_solution(self, X_upper):
        d_diff = self.d_lower - self.d_common
        assert d_diff - 1 > 0, "SMD10: d_lower - d_common should be strictly greater than 1."
        return np.c_[
            np.ones((X_upper.shape[0], d_diff)) / np.sqrt(d_diff - 1), 
            np.arctan(X_upper[:, -self.d_common:])]

    def _F1(self, X_u1):
        return ((X_u1 - 2)**2).sum(axis=1)

    def _F2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return ((X_u2 - 2)**2).sum(axis=1) - ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return ((X_l1 - 2)**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)
    
    def set_constraint_num(self):
        self.upper_constraint_num = self.d_upper
        self.lower_constraint_num = self.d_lower - self.d_common
    
    def upper_constraint(self, X_upper, X_lower):
        # X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        # X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        return X_upper + X_upper**3 - (X_upper**3).sum(axis=1)[None,:]

    def lower_constraint(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        # X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        return X_l1 + X_l1**3 - (X_l1**3).sum(axis=1)[None,:]


class SMD11(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[- 5 * np.ones(d_diff), - 1 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 1 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), np.ones(self.d_common) / np.e]
        lower_domain_max = np.r_[10 * np.ones(d_diff), np.e * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.zeros(self.d_upper)

    def lower_optimal_solution(self, X_upper):
        return None

    def is_lower_optimal_solution(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        flag1 = np.abs(X_l1).sum(axis=1) == 0
        # flag2 = ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1) == 1 # this is affected by numerical errors
        eps = 1e-12
        flag2 = np.abs(((X_u2 - np.log(X_l2)) ** 2).sum(axis=1) - 1) < eps
        return np.logical_and(flag1, flag2)

    def lower_optimal_solution_with_upper_opt(self):
        d_diff = self.d_lower - self.d_common
        return np.r_[np.zeros(d_diff), np.exp(-1/np.sqrt(self.d_common)) * np.ones(self.d_common)]

    def _F1(self, X_u1):
        return (X_u1**2).sum(axis=1)

    def _F2(self, X_l1):
        return - (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return (X_u2**2).sum(axis=1) - ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1)
    
    def set_constraint_num(self):
        self.upper_constraint_num = self.d_common
        self.lower_constraint_num = 1
        
    def upper_constraint(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        return X_u2 - (1 / np.sqrt(self.d_common) + np.log(X_l2))

    def lower_constraint(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        return ((X_u2 - np.log(X_l2)) ** 2).sum(axis=1) - 1


class SMD12(SMD):
    def upper_domain(self):
        d_diff = self.d_upper - self.d_common
        upper_domain_min = np.r_[- 5 * np.ones(d_diff), - 14.10 * np.ones(self.d_common)]
        upper_domain_max = np.r_[10 * np.ones(d_diff), 14.10 * np.ones(self.d_common)]
        return upper_domain_min, upper_domain_max

    def lower_domain(self):
        d_diff = self.d_lower - self.d_common
        lower_domain_min = np.r_[- 5 * np.ones(d_diff), -1.5 * np.ones(self.d_common)]
        lower_domain_max = np.r_[10 * np.ones(d_diff), 1.5 * np.ones(self.d_common)]
        return lower_domain_min, lower_domain_max
    
    def upper_optimal_solution(self):
        return np.ones(self.d_upper) / np.sqrt(self.d_upper - 1)

    def lower_optimal_solution(self, X_upper):
        return None

    def is_lower_optimal_solution(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        d_diff = self.d_lower - self.d_common
        flag1 = np.abs(X_l1 - 1/np.sqrt(d_diff - 1)).sum(axis=1) == 0
        # flag2 = ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1) == 1 # this is affected by numerical errors
        eps = 1e-12
        flag2 = np.abs(((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1) - 1) < eps
        return np.logical_and(flag1, flag2)

    def lower_optimal_solution_with_upper_opt(self):
        # this differs from the solution below (51) in https://www.egr.msu.edu/~kdeb/papers/k2013002.pdf
        # refer https://github.com/jmejia8/bilevel-benchmark/blob/05437a48a1ee0fd5c2c464e8affc6994b4136582/smd.c#L157
        d_diff = self.d_lower - self.d_common
        return np.r_[
            np.ones(d_diff) / np.sqrt(d_diff - 1), 
            np.arctan(1 / np.sqrt(self.d_upper - 1) - 1 / np.sqrt(self.d_common)) * np.ones(self.d_common)]

    def _F1(self, X_u1):
        return ((X_u1 - 2)**2).sum(axis=1)

    def _F2(self, X_l1):
        return (X_l1**2).sum(axis=1)

    def _F3(self, X_u2, X_l2):
        return ((X_u2 - 2)**2).sum(axis=1) \
            + np.tan(np.abs(X_l2)).sum(axis=1) \
            - ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)

    def _f1(self, X_u1, X_u2):
        return (X_u1**2).sum(axis=1)

    def _f2(self, X_l1):
        return ((X_l1 - 2)**2).sum(axis=1)

    def _f3(self, X_u2, X_l2):
        return ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)

    def set_constraint_num(self):
        d_diff = self.d_lower - self.d_common
        self.upper_constraint_num = self.d_upper + self.d_common
        self.lower_constraint_num = 1 + d_diff
        
    def upper_constraint(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        constraint_1 = X_u2 - np.tan(X_l2)
        constraint_2 = X_upper + X_upper**3 - (X_upper**3).sum(axis=1)[None,:]
        return np.c_[constraint_1, constraint_2]

    def lower_constraint(self, X_upper, X_lower):
        X_l1, X_l2 = X_lower[:, :-self.d_common], X_lower[:, -self.d_common:]
        X_u1, X_u2 = X_upper[:, :-self.d_common], X_upper[:, -self.d_common:]
        constraint_1 = ((X_u2 - np.tan(X_l2)) ** 2).sum(axis=1)[:, None] - 1
        constraint_2 = X_l1 + X_l1**3 - (X_l1**3).sum(axis=1)[None,:]
        return np.c_[constraint_1, constraint_2]


