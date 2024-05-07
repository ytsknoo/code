#!/usr/bin/env python
# -*- coding: utf-8 -*-

from tabnanny import verbose
import numpy as np
import os, sys
from GPyOpt.methods import BayesianOptimization
from GPyOpt import Design_space
from GPyOpt.core import BO
from GPyOpt.core.evaluators import Sequential
from GPyOpt.models import GPModel
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionLCB
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.experiment_design.random_design import RandomDesign
import matplotlib.pyplot as plt
from GPy.kern import Kern, RBF
from GPy.models import GPRegression


sys.path.append(os.path.dirname(__file__) + "/../../")
from src.objective_function import benchmark as bench
from src.model.gp import gp_interface as gp
from src.model.gp import bo_acq
from src.util import sampler as sampler
from src.util import weight as weight

current_dir = os.path.dirname(__file__)

def gp_run():

    def f(X):
        # return np.sum(X * X, axis=1)[:, np.newaxis] 
        d = X.shape[1]
        coef = 10 ** (np.arange(d) / float(d - 1))
        _X = X * coef
        return np.c_[
            np.sum(X * X, axis=1), # + np.random.rand(X.shape[0]), 
            - np.sum(X * X, axis=1), # + np.random.rand(X.shape[0]), 
            np.sum(_X * _X, axis=1), # + np.random.rand(X.shape[0]), 
            - np.sum(_X * _X, axis=1), # + np.random.rand(X.shape[0]),
            #np.sum(np.abs(X), axis=1),
            #- np.sum(np.abs(X), axis=1)
            ]

    d = 3 # num of dim
    n = 10 # num of data

    X = np.random.rand(n, d) * 4 - 2
    Y = f(X)
    
    gp_interface = gp.GPInterface(d)
    gp_interface._create_model(np.array([X for _ in range(Y.shape[1])]), np.array([Y]).T)

    Z = np.ones((2, d)) + np.random.randn(2, d)
    print("input: {}".format(Z))
    print("true_eval: {}".format(f(Z)))
    print("predict: {}".format(gp_interface.predict_mean(Z).T))
    print("predict_sigma: {}".format(gp_interface.predict_sigma(Z)))
    print("predict_covariance: {}".format(gp_interface.predict_covariance(Z)))


    if d == 1:
        # plot result
        fig, ax = plt.subplots()
        if Y.shape[1] == 1:
            gp_interface.model.plot(ax=ax)
        else:
            for i in range(Y.shape[1]):
                gp_interface.model.plot(fignum=1,fixed_inputs=[(1, i)],ax=ax,legend=i==0)
        plt.xlabel('x')
        plt.ylabel('y')
        fig.savefig(os.path.join(current_dir, "gp_test.pdf"))


def bo_run():
    # problem definition
    d = 3
    f = bench.ShiftSphere(d, param=0)
    domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': (-2,2)} for i in range(d)]

    feasible_region = Design_space(space = domain)
    model = GPModel(exact_feval=False, optimize_restarts=0, verbose=False)
    aquisition_optimizer = AcquisitionOptimizer(feasible_region)

    # var_acquisition = bo_acq.AcquisitionVariance(model, feasible_region, optimizer=aquisition_optimizer)
    # var_acquisition = AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)

    bo = BayesianOptimization(model=model, f=f, domain=domain, acquisition_type="EI", initial_design_numdata = d+1)
    bo.run_optimization(max_iter=100)

    print("best solution: {}".format(bo.x_opt))
    print("best evaluation: {}".format(bo.fx_opt))

    Z = np.zeros((1, d))
    predict_m, predict_st = bo.model.predict(Z)
    print("input: {}".format(Z))
    print("true_eval: {}".format(f(Z)))
    print("predict(mean,std): {}, {}".format(predict_m, predict_st))

    #bo.plot_acquisition(filename=os.path.join(current_dir, "acquisition.pdf"))
    bo.plot_convergence(filename=os.path.join(current_dir, "convergence.pdf"))


def bo_step_run():
    # problem definition
    d = 3
    f = bench.ShiftSphere(d, param=0)
    domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': (-2,2)} for i in range(d)]

    feasible_region = Design_space(space = domain)

    # init sample
    init_sample_num = d+2
    init_design = RandomDesign(feasible_region)
    X = init_design.get_samples(init_sample_num)
    Y = np.array([f(x) for x in X])

    aquisition_optimizer = AcquisitionOptimizer(feasible_region)
    model = GPModel(exact_feval=True, optimize_restarts=0, verbose=False)
    
    for i in range(30):
        model.updateModel(X,Y,None,None)
        acquisition = AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
        bo = BayesianOptimization(f=None, domain=domain, acquisition=acquisition, X=X, Y=Y, exact_feval=True)
        
        x_suggest = bo.suggest_next_locations()
        y = f(x_suggest)
        X = np.vstack((X, x_suggest))
        Y = np.vstack((Y, y))

        # print("best solution: {}".format(f.best_solution))
        print("({}) best evaluation: {}".format(i,f.best_eval))

    Z = np.zeros((1, d))
    predict_m, predict_st = bo.model.predict(Z)
    print("input: {}".format(Z))
    print("true_eval: {}".format(f(Z)))
    print("predict(mean,std): {}, {}".format(predict_m, predict_st))


def bo_step_run_with_interface():
    # problem definition
    d = 3
    f = bench.ShiftSphere(d, param=0)
    domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': (-2,2)} for i in range(d)]

    feasible_region = Design_space(space = domain)

    # init sample
    init_sample_num = d+2
    init_design = RandomDesign(feasible_region)
    init_sample = init_design.get_samples(init_sample_num)
    init_evaluations = np.array([f(x) for x in init_sample])

    model = gp.GPInterface(f.d, exact_feval=True, optimize_restarts=0, verbose=False)
    model._create_model(init_sample, init_evaluations)
    # model = GPModel(exact_feval=False, optimize_restarts=0, verbose=False)
    aquisition_optimizer = AcquisitionOptimizer(feasible_region)

    for i in range(30):
        # acquisition = bo_acq.AcquisitionVariance(model, feasible_region, optimizer=aquisition_optimizer)
        acquisition = AcquisitionEI(model, feasible_region, optimizer=aquisition_optimizer)
        bo = BayesianOptimization(f=None, domain=domain, acquisition=acquisition, X=model.X_data, Y=model.Y_data, exact_feval=True, maximize=False)
        
        x_suggest = bo.suggest_next_locations()
        y = f(x_suggest)
        model.add_data(x_suggest, y)

        # print("best solution: {}".format(f.best_solution))
        print("({}) best evaluation: {}".format(i,f.best_eval))

    Z = np.zeros((1, d))
    predict_m, predict_st = model.predict_mean(Z), model.predict_sigma(Z)
    print("input: {}".format(Z))
    print("true_eval: {}".format(f(Z)))
    print("predict(mean,std): {}, {}".format(predict_m, predict_st))

if __name__ == '__main__':
    # gp_run()
    # bo_run()
    # bo_step_run()
    bo_step_run_with_interface()
