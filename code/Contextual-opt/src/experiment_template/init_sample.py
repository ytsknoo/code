from pickletools import optimize
import numpy as np
import pandas as pd
import os, sys
import scipy.linalg
import matplotlib.pyplot as plt

import copy
import pickle

sys.path.append(os.path.dirname(__file__) + "/../../")
sys.path.append(os.path.dirname(__file__))
from src.objective_function.base import ParametrizedObjectiveFunction
from src.objective_function import benchmark as bench
from src.optimizer.base_optimizer import BaseOptimizer
from src.optimizer import cmaes as cma
from src.util import sampler as sampler
from src.util import weight as weight
from src.model.gp import bo_acq
from src.model.gp import gp_interface as gp
from experiment_train import ParameterFunction
import experiment_train

from GPy.kern import Kern, RBF, Bias, Linear, Coregionalize, Matern52, Matern32
from GPy.util.multioutput import LCM, ICM
from GPyOpt import Design_space
from GPyOpt.experiment_design.random_design import RandomDesign

class InitSample(object):
    # d:次元数
    # f:目的関数
    # init_saple_num:初期サンプル数

    def __init__(self,d,f,init_sample_num,max_eval):
        self.d = d
        self.f = f
        self.init_sample_num = init_sample_num

        self.max_eval = max_eval
        self.init_sample = None
        self.best_solutions = None
        self.x_log = None
        self.eval_log = None
        
    
    def init_optimize(self,verbose=True):
        # target step-size for convergence
        # target_std = 1e-12
        target_std = 1e-10

        """
        set optimizer
        """
        init_m, init_sigma = np.random.rand(self.d) * 2 - 1, 2
        lam = cma.CMAParam.pop_size(self.d)
        w_func = weight.CMAWeight(lam, min_problem=self.f.minimization_problem)
        _optimizer = cma._CMAES(self.d, w_func, m=init_m, sigma=init_sigma,fin_cov=True)
        _optimizer.set_terminate_condition(target_std)

        """
        init sample
        """
        domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': self.f.param_range[i]} for i in range(self.f.param_dim)]
        feasible_region = Design_space(space = domain)

        # ランダムに文脈ベクトル決定
        init_design = RandomDesign(feasible_region)
        self.init_sample = init_design.get_samples(self.init_sample_num) # ここをランダムに変更すればいい(今すでにランダムかも)
        self.best_solutions = np.zeros((self.init_sample_num, self.d))
        # print(self.init_sample)

        restart = True

        for i, param in enumerate(self.init_sample):
            self.f_param = ParameterFunction(self.f, max_eval=self.max_eval)
            while True:
                optimizer = copy.deepcopy(_optimizer)
                self.f_param.set_optimizer(optimizer)
                self.best_solutions[i] = self.f_param(param,noclear_fcall=True)
                # self.best_solutions[i] = self.f.optimal_solution(param)
                if (not restart) or (self.f._evaluation([self.best_solutions[i]],self.init_sample[i]) < 1e-07) or (self.f_param.eval_count >= self.max_eval):
                    break
            print('init_sample({})'.format(i))
            print('Param:{}'.format(self.init_sample[i]))
            print('Best_solution:{}'.format(self.best_solutions[i]))
            print('Eval:{}'.format(self.f._evaluation([self.best_solutions[i]],self.init_sample[i])))
            print('Fcalls:{}'.format(self.f_param.eval_count))


        # print(self.best_solutions)

        if verbose:
            print("finished design init (init_ite:{}, f-calls: {})".format(self.init_sample_num, self.f_param.eval_count))

        if self.f_param.eval_count >= self.max_eval+100:
            print("||Error|| upper limit of number of evaluations is too small (f-calls: {}, max_eval: {})".format(self.f_param.eval_count, self.max_eval))
            exit()

        assert not np.isnan(self.best_solutions).any(), self.best_solutions
        assert not np.isnan(self.init_sample).any(), self.init_sample
    
    def get_init_param(self):
        return self.init_sample
    
    def get_best_solutions(self):
        return self.best_solutions
    
