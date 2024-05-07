#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../../")
from src.objective_function import benchmark as bench
from src.optimizer import cmaes as cma
from src.util import sampler as sampler
from src.util import weight as weight


def cma_run():
    # problem definition
    d = 20
    f = bench.ShiftSphere(d, max_eval=1e3, param=2)
    init_m, init_sigma = np.ones(d), 2
    lam = cma.CMAParam.pop_size(d)
    # weight function
    w_func = weight.CMAWeight(lam, min_problem=f.minimization_problem)
    # optimizer
    opt = cma.CMAES(d, w_func, m=init_m, sigma=init_sigma)
    # run
    return opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)

def bc_cma_run():
    # problem definition
    d = 8
    f = bench.ShiftSphere(d, max_eval=1e5, param=2)
    upper_bound = np.ones(d) * 8
    lower_bound = np.ones(d) * 4
    
    init_m, init_sigma = np.arange(d) * 6, 2
    lam = cma.CMAParam.pop_size(d)
    # weight function
    w_func = weight.CMAWeight(lam, min_problem=f.minimization_problem)
    # optimizer
    opt = cma._BoxConstraintCMAES(d, w_func, upper_bound, lower_bound, m=init_m, sigma=init_sigma)
    # run
    opt.run(sampler.DefaultSampler(f, lam), logger=None, verbose=True)

    return f.best_solution

if __name__ == '__main__':
    print(cma_run())
    # print(bc_cma_run())

