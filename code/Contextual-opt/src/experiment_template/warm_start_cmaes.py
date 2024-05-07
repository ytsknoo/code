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
from cmaes import CMA, get_warm_start_mgd

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

class WarmStartCmaes (object):
    # d:次元数
    # f:目的関数
    # init_saple:初期サンプルパラメータ
    # best_solutions:初期パラメータに対応する最適解
    # target_param:最適解を求めたいパラメータ

    def __init__(self,d,f,init_sample,best_solutions,max_eval,log_name=None,log_path="./"):
        self.d = d
        self.f = f
        self.init_sample = init_sample
        self.best_solutions = best_solutions

        # self.max_eval = 1e3 * d
        self.max_eval = max_eval
        # self.f_param = ParameterFunction(f, self.max_eval,log_name=log_name,log_path=log_path)
        self.best_solution = None
        self.log_path = log_path
        self.log_name = log_name

    
    def optimize(self,target_param,logname=None,verbose=True):
        self.target_param = target_param

        # target step-size for convergence
        target_std = 1e-6
        # target_std = 1e-20

        min_target_init_dis = np.inf
        min_target_init_num = 0

        # target_paramと近いベクトル決定
        for i in range(len(self.init_sample)):
            if np.linalg.norm(target_param-self.init_sample[i]) < min_target_init_dis:
                min_target_init_dis = np.linalg.norm(target_param-self.init_sample[i])
                min_target_init_num = i
        
        
        # warmstart
        source_solutions = []
        for _ in range(int(self.max_eval)):
            x =  np.random.random(self.d)*4 - 2
            eval = self.f._evaluation(np.array([x]),self.init_sample[min_target_init_num])
            source_solutions.append((x, eval))
        
        ws_mean, ws_sigma, ws_cov = get_warm_start_mgd(
        source_solutions, gamma=0.1, alpha=0.1
        )

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

        # 最適化
        optimizer = copy.deepcopy(_optimizer)
        optimizer.set_model_param(mean=ws_mean, std=ws_sigma, cov=ws_cov)
        if evo_path is not None:
            optimizer.set_evolution_path(*evo_path)
        self.f_param.set_optimizer(optimizer)
        self.best_solution = self.f_param(self.target_param)

        assert not np.isnan(self.best_solution).any(), 'param:{}'.format(self.target_param)

        # model.add_data(np.array([self.target_param[0] for _ in range(self.f.d)]), np.array([self.best_solution]).T)

        if verbose:
            print("WS-CMA-ES")
            print("||Success|| optimization finished (f-calls: {})".format(self.f_param.eval_count))

        return self.f._evaluation(np.array([self.best_solution]),self.target_param)
    
    def get_best_solution(self):
        return self.best_solution