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

class ContextualGausianCmaesAcc (object):
    # d:次元数
    # f:目的関数
    # init_saple:初期サンプルパラメータ
    # best_solutions:初期パラメータに対応する最適解
    # target_param:最適解を求めたいパラメータ

    def __init__(self,d,f,init_sample,best_solutions,log_name=None,log_path="./"):
        self.d = d
        self.f = f
        self.init_sample = init_sample
        self.best_solutions = best_solutions

        self.max_eval = 1e3 * d
        # self.f_param = ParameterFunction(f, self.max_eval,log_name=log_name,log_path=log_path)
        self.best_solution = None
        self.log_path = log_path
        self.log_name = log_name

    
    def optimize(self,target_param,logname=None,verbose=True):
        self.target_param = target_param

        # hyperparameters for initial distribution of CMA-ES
        max_std_init = 2
        min_std_init = 1e-2
        # min_std_init = 1e-10
        max_eigval_cov_init = 1e3
        min_eigval_cov_init = 1e-3

        # target step-size for convergence
        target_std = 1e-6
        # target_std = 1e-20

        """
        set optimizer
        """
        init_m, init_sigma = np.random.rand(self.d) * 2 - 1, 2
        lam = cma.CMAParam.pop_size(self.d)
        w_func = weight.CMAWeight(lam, min_problem=self.f.minimization_problem)
        _optimizer = cma._CMAES(self.d, w_func, m=init_m, sigma=init_sigma,sampling_mean=False)
        _optimizer.set_terminate_condition(target_std)


        evo_path = None

        if logname:
            self.log_name = logname
            
        self.f_param = ParameterFunction(self.f, self.max_eval,log_name=self.log_name,log_path=self.log_path)


        # cmaes初期化
        y_mean = self.model.predict_mean(self.target_param)
        y_std_diag = self.model.predict_sigma(self.target_param)
        y_std = np.sqrt(np.sum(y_std_diag ** 2) / self.d)
        y_cov = self.model.predict_covariance(self.target_param)[0] / (y_std ** 2)

        # # 単位行列で初期化
        y_cov = np.eye(self.f.d)

        assert not np.isnan(y_mean).any()
        assert not np.isnan(y_std).any()
        assert not np.isnan(y_cov).any()

        # refine initial distritbution parameter
        if np.isnan(y_cov).any():
            y_cov = np.eye(self.f.d)
        else: # 固有値でクリップするために一度分解して戻している？
            eigvals, B = scipy.linalg.eigh(y_cov) #固有値eigvals 固有ベクトルB
            eigvals = np.clip(eigvals, min_eigval_cov_init, max_eigval_cov_init)
            y_cov = np.dot(np.dot(B, np.diag(eigvals)), B.T)
        y_std = np.clip(y_std, min_std_init, max_std_init) #ここのせいでσが大きくなっている
        # min_std_initを小さくすると先に終了条件を満たしてしまって最適化が行われない

        # 最適化
        optimizer = copy.deepcopy(_optimizer)
        optimizer.set_model_param(mean=y_mean.flatten(), std=y_std, cov=y_cov)
        if evo_path is not None:
            optimizer.set_evolution_path(*evo_path)
        self.f_param.set_optimizer(optimizer)
        self.best_solution = self.f_param(self.target_param)

        assert not np.isnan(self.best_solution).any(), 'param:{}'.format(self.target_param)

        # model.add_data(np.array([self.target_param[0] for _ in range(self.f.d)]), np.array([self.best_solution]).T)

        if verbose:
            print("Proposed")
            print("||Success|| optimization finished (f-calls: {})".format(self.f_param.eval_count))
            # print("Param:{}".format(self.target_param))

        return self.f._evaluation(np.array([self.best_solution]),self.target_param)
    
    def create_model(self):
        """
        Bayes optimization
        """
        # モデル作成
        kernels_list = [RBF(self.f.param_dim), Linear(self.f.param_dim), Matern52(self.f.param_dim)]
        kernel = LCM(input_dim=self.f.param_dim, num_outputs=self.f.d, kernels_list=kernels_list)

        self.model = gp.GPInterface(self.f.param_dim, kernel=kernel, exact_feval=True, optimize_restarts=0, verbose=False)
        self.model._create_model(np.array([self.init_sample for _ in range(self.f.d)]), np.array([self.best_solutions]).T) # 既存の最適解からモデル作成ここまで
        return
    
    def get_best_solution(self):
        return self.best_solution