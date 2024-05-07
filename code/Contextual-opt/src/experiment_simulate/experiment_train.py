from pickletools import optimize
from turtle import clear
import numpy as np
import pandas as pd
import os, sys
import scipy.linalg
from paramz import ObsAr

from GPyOpt.methods import BayesianOptimization
from GPyOpt import Design_space
from GPyOpt.models import GPModel
from GPyOpt.optimization import AcquisitionOptimizer
from GPyOpt.experiment_design.random_design import RandomDesign
from GPy.kern import Kern, RBF, Bias, Linear, Coregionalize, Matern52, Matern32
from GPy.util.multioutput import LCM, ICM

import copy
import pickle

sys.path.append(os.path.dirname(__file__) + "/../../")
from src.objective_function.base import ParameterFunction
from src.objective_function import benchmark as bench
from src.optimizer import cmaes as cma
from src.util import sampler as sampler
from src.util import weight as weight
from src.model.gp import bo_acq
from src.model.gp import gp_interface as gp


"""
simplified evaluation
"""
def simplified_evaluation(f, feasible_region, model, evaluation_sample_num=100):
    l2_dist = np.ones(evaluation_sample_num) * np.inf
    evals_diff = np.ones(evaluation_sample_num) * np.inf

    evaluation_design = RandomDesign(feasible_region)
    evaluation_sample = evaluation_design.get_samples(evaluation_sample_num)

    for i, param in enumerate(evaluation_sample):
        true_solution = f.optimal_solution(param)
        true_best_eval = f.optimal_evaluation(param)
        if true_solution is None:
            # run optimizer until maximum evaluation
            init_m, init_sigma = np.random.rand(f.d) * 2 - 1, 2
            lam = cma.CMAParam.pop_size(f.d)
            w_func = weight.CMAWeight(lam, min_problem=f.minimization_problem)
            optimizer = cma._CMAES(f.d, w_func, m=init_m, sigma=init_sigma)
            optimizer.set_terminate_condition(1e-12)
            f.clear()
            f.max_eval = 1e4 * f.d
            optimizer.run(sampler.DefaultSampler(f, optimizer.lam), logger=None, verbose=False)
            true_solution = f.best_solution
        
        predicted_solution = model.predict_mean(param)[0]
        l2_dist[i] = np.sum((predicted_solution - true_solution) ** 2)
        evals_diff[i] = f.evaluation(predicted_solution, param) - f.optimal_evaluation(param)

        assert np.isclose(f.evaluation(true_solution, param)[0], true_best_eval), f.evaluation(true_solution, param)[0]
        assert evals_diff[i] >= 0, evals_diff[i]
    return {"l2_mean": np.mean(l2_dist), "eval_diff_mean":np.mean(evals_diff)}


"""
experiment
"""
def experiment_train(d,f,verbose=True, running_evaluation=True):

    """
    problem settings
    """
    # d = 2
    max_eval = 1e3 * d
    # f = bench.ShiftSphere(d)
    # f = bench.ShiftEllipsoid(d)
    f_param = ParameterFunction(f, max_eval=max_eval)
    init_sample_num = d

    # hyperparameters for initial distribution of CMA-ES
    max_std_init = 2
    min_std_init = 1e-2
    max_eigval_cov_init = 1e3
    min_eigval_cov_init = 1e-3

    # target step-size for convergence
    target_std = 1e-6

    """
    set optimizer
    """
    init_m, init_sigma = np.random.rand(d) * 2 - 1, 2
    lam = cma.CMAParam.pop_size(d)
    w_func = weight.CMAWeight(lam, min_problem=f.minimization_problem)
    _optimizer = cma._CMAES(d, w_func, m=init_m, sigma=init_sigma)
    _optimizer.set_terminate_condition(target_std)

    """
    init sample
    """
    domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': f.param_range[i]} for i in range(f.param_dim)]
    feasible_region = Design_space(space = domain)

    # ベイズ最適化の獲得関数でパラメータ取得？
    init_design = RandomDesign(feasible_region)
    init_sample = init_design.get_samples(init_sample_num) # ここをランダムに変更すればいい(今すでにランダムかも)
    best_solutions = np.zeros((init_sample_num, d))
    print(init_sample)

    for i, param in enumerate(init_sample):
        optimizer = copy.deepcopy(_optimizer)
        f_param.set_optimizer(optimizer)
        best_solutions[i] = f_param(param)

    print(best_solutions)

    if verbose:
        print("finished design init (init_ite:{}, f-calls: {})".format(init_sample_num, f_param.eval_count))

    if f_param.eval_count >= max_eval:
        print("||Error|| upper limit of number of evaluations is too small (f-calls: {}, max_eval: {})".format(f_param.eval_count, max_eval))
        exit()

    assert not np.isnan(best_solutions).any(), best_solutions
    assert not np.isnan(init_sample).any(), init_sample

    """
    Bayes optimization
    """
    kernels_list = [RBF(f.param_dim), Linear(f.param_dim), Matern52(f.param_dim)]
    kernel = LCM(input_dim=f.param_dim, num_outputs=f.d, kernels_list=kernels_list)

    model = gp.GPInterface(f.param_dim, kernel=kernel, exact_feval=True, optimize_restarts=0, verbose=False)
    model._create_model(np.array([init_sample for _ in range(f.d)]), np.array([best_solutions]).T) # 既存の最適解からモデル作成ここまで

    # aquisition_optimizer = AcquisitionOptimizer(feasible_region)

    # ite = init_sample_num
    evo_path = None
    # var_acquisition = bo_acq.AcquisitionVariance(model, feasible_region, optimizer=aquisition_optimizer)
    # bo = BayesianOptimization(f=None, domain=domain, acquisition=var_acquisition, normalize_Y=False, X=model.X_data, Y=model.Y_data, )

    # assert not np.isnan(model.X_data).any()
    # assert not np.isnan(model.Y_data).any()

    f_param.eval_count = 0
    
    param_suggest = init_design.get_samples(1)
    print(param_suggest)
    y_mean = model.predict_mean(param_suggest)
    y_std_diag = model.predict_sigma(param_suggest)
    y_std = np.sqrt(np.sum(y_std_diag ** 2) / d)
    y_cov = model.predict_covariance(param_suggest)[0] / (y_std ** 2)

    assert not np.isnan(y_mean).any()
    assert not np.isnan(y_std).any()
    assert not np.isnan(y_cov).any()

    # refine initial distritbution parameter
    if np.isnan(y_cov).any():
        y_cov = np.eye(f.d)
    else: # 固有値でクリップするために一度分解して戻している？
        eigvals, B = scipy.linalg.eigh(y_cov) #固有値eigvals 固有ベクトルB
        eigvals = np.clip(eigvals, min_eigval_cov_init, max_eigval_cov_init)
        y_cov = np.dot(np.dot(B, np.diag(eigvals)), B.T)
    y_std = np.clip(y_std, min_std_init, max_std_init)

    optimizer = copy.deepcopy(_optimizer)
    optimizer.set_model_param(mean=y_mean.flatten(), std=y_std, cov=y_cov)
    if evo_path is not None:
        optimizer.set_evolution_path(*evo_path)
    f_param.set_optimizer(optimizer)
    best_solution = f_param(param_suggest)

    assert not np.isnan(best_solution).any(), 'param:{}'.format(param_suggest)

    model.add_data(np.array([param_suggest[0] for _ in range(f.d)]), np.array([best_solution]).T)

    # save evolution path
    evo_path = optimizer.get_evolution_path()

    if verbose:
        print_sentence = "f-calls: {}".format( f_param.eval_count)
        if running_evaluation:
            result = simplified_evaluation(f, feasible_region, model, evaluation_sample_num=100)
            print_sentence += ", \t mean L2_dist: {}".format(result["l2_mean"])
        print(print_sentence)

    # ite +=1

    if verbose:
        print("||Success|| optimization finished (f-calls: {})".format(f_param.eval_count))

    # save trained model
    # current_dir = os.path.dirname(__file__)
    # np.save(current_dir + '/model_save.npy', model.model.param_array) #モデルのパラメータデータ？
    # np.save(current_dir + '/model_data_X.npy', model.X_data) #文脈ベクトル？
    # np.save(current_dir + '/model_data_Y.npy', model.Y_data) #最適解？

    print(model.X_data)
    print(model.Y_data)

    # if verbose:
    #     print("||Success|| model is saved at {}.pickle".format(current_dir))

    """
    evaluation (simplified)
    """
    # evaluation_sample_num = 1000
    # result = simplified_evaluation(f, feasible_region, model, evaluation_sample_num=evaluation_sample_num)
    # print("||Evaluation (n={})||".format(evaluation_sample_num))
    # print("\t mean L2 distance: {}".format(result["l2_mean"]))
    # print("\t mean evaluation difference: {}".format(result["eval_diff_mean"]))

    return

if __name__ == '__main__':
    experiment_train(2,bench.ShiftSphere(2)) 