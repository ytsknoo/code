#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import os, sys

sys.path.append(os.path.dirname(__file__) + "/../../")
from src.objective_function import smd_benchmark as smd

def evaluation_test():
    # problem definition
    d_upper = 5
    d_lower = 10
    f = smd.SMD12(d_upper, d_lower)

    # confirm domain
    upper_domain_min, upper_domain_max = f.upper_domain()
    lower_domain_min, lower_domain_max = f.lower_domain()
    print("domains for upper and lower variables")
    print("min and max of domain for upper variable:{}, {}".format(upper_domain_min, upper_domain_max))
    print("min and max of domain for lower variable:{}, {}".format(lower_domain_min, lower_domain_max))

    # confirm optimal solution 
    upper_opt = f.upper_optimal_solution()
    lower_opt = f.lower_optimal_solution_with_upper_opt()[None,:]
    print("upper and lower optimal solution")
    print("upper opt. solution: {}".format(upper_opt))
    print("lower opt. solution: {}".format(lower_opt))

    # confirm evaluation value at optimal solution 
    upper_opt_eval = f._evaluation_upper(upper_opt[None,:], lower_opt)
    lower_opt_eval = f._evaluation_lower(upper_opt[None,:], lower_opt)
    print("evaluation values at upper and lower optimal solution (should be zero)")
    print("f_upper({}, {}) = {}".format(upper_opt[None,:], lower_opt, upper_opt_eval))
    print("f_lower({}, {}) = {}".format(upper_opt[None,:], lower_opt, lower_opt_eval))

    # confirm other evaluation value at other solution 
    upper_solution = np.ones(d_upper)[None,:]
    lower_solution = np.ones(d_lower)[None,:]
    upper_eval = f._evaluation_upper(upper_solution, lower_solution)
    lower_eval = f._evaluation_lower(upper_solution, lower_solution)
    print("evaluation values at other solution")
    print("f_upper({}, {}) = {}".format(upper_solution, lower_solution, upper_eval))
    print("f_lower({}, {}) = {}".format(upper_solution, lower_solution, lower_eval))

    # confirm constraints
    print("constraint values at optimal solution (violated if constraint value is negative)")
    print(f.upper_constraint(upper_opt[None,:], lower_opt))
    print(f.lower_constraint(upper_opt[None,:], lower_opt))

    print("constraint values at other solution (violated if constraint value is negative)")
    print(f.upper_constraint(upper_solution, lower_solution))
    print(f.lower_constraint(upper_solution, lower_solution))

    # confirm other functions
    print("check for \"is_lower_optimal_solution\"")
    print(f.is_lower_optimal_solution(upper_opt[None,:], lower_opt))
    print(f.is_lower_optimal_solution(upper_opt[None,:], lower_solution))


if __name__ == '__main__':
    evaluation_test()
