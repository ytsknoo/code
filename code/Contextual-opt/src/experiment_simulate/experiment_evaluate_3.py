from pickletools import optimize
import numpy as np
import pandas as pd
import os, sys
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib

import copy
import pickle

sys.path.append(os.path.dirname(__file__) + "/../../")
sys.path.append(os.path.dirname(__file__))
from src.objective_function.base import ParametrizedObjectiveFunction
from src.objective_function import fech_push as bench
from src.optimizer.base_optimizer import BaseOptimizer
from src.optimizer import cmaes as cma
from src.util import sampler as sampler
from src.util import weight as weight
from src.model.gp import bo_acq
from src.model.gp import gp_interface as gp
from experiment_train import ParameterFunction
from init_sample import InitSample
from contextual_gausian_cmaes_acc import ContextualGausianCmaesAcc
from warm_start_cmaes import WarmStartCmaes


from GPy.kern import Kern, RBF, Bias, Linear, Coregionalize, Matern52, Matern32
from GPy.util.multioutput import LCM, ICM
from GPyOpt import Design_space
from GPyOpt.experiment_design.random_design import RandomDesign

import math

# シード設定
np.random.seed(47)

"""
experiment
"""
def experiment_evaluate(experiment_times=1,log_path=None,verpose=True,verbose=True):
    fname = "FetchPush"

    f = bench.FechPush(log_video=True)
    novideo_f = bench.FechPush(log_video=False)

    d = f.d
    max_eval = 5e2
    
    init_sample_num = 10
    target_param_num = 1

    # target_param決定
    # domain = [{'name': 'var_{}'.format(i), 'type': 'continuous', 'domain': f.param_range[i]} for i in range(f.param_dim)]
    # feasible_region = Design_space(space = domain)

    # target_design = RandomDesign(feasible_region)
    # target_param = target_design.get_samples(1)

    target_param = np.zeros((target_param_num,len(f.param_range)))


    experiment_evals_suggest = np.zeros((experiment_times,target_param_num))
    experiment_evals_cmaes = np.zeros((experiment_times,target_param_num))
    experiment_evals_wscmaes = np.zeros((experiment_times,target_param_num))

    # 最適化
    # 提案手法
    for i in range(experiment_times*2,experiment_times*3):

        # 初期サンプル生成
        init = InitSample(d,novideo_f,init_sample_num,max_eval)

        init.init_optimize()
        init_param = init.get_init_param()
        best_solutions = init.get_best_solutions()
        delete_list = np.array([])
                


        # ターゲットパラメータ決定
        param_range = np.array(f.param_range)
        for j in range(target_param_num):
            target_param[j,:] = np.random.random((1,len(param_range)))*(param_range[:,1]-param_range[:,0]).T+param_range[:,0].T

        
        ConGauCma = ContextualGausianCmaesAcc(d,f,init_param,best_solutions,max_eval=max_eval,log_path=log_path)
        ConGauCma.create_model()
        for k in range(target_param_num):
            f.clear()
            f.set_video_name("{}/CMAES-CWS_{}_{}".format(fname,i,k))
            ConGauCma.optimize(np.array([target_param[k,:]]),logname="c_gausian_cmaes_log_{}_{}.csv".format(i,k))

        WSCmaes = WarmStartCmaes(d,f,init_param,best_solutions,max_eval=max_eval,log_path=log_path)
        for k in range(target_param_num):
            f.clear()
            f.set_video_name("{}/WS-CMAES_{}_{}".format(fname,i,k))
            WSCmaes.optimize(np.array([target_param[k,:]]),logname="ws_cmaes_log_{}_{}.csv".format(i,k))

        for k in range(target_param_num):
            f.clear()
            f.set_video_name("{}/CMAES_{}_{}".format(fname,i,k))
            # CMA-ES
            cma_f_param = ParameterFunction(f, max_eval=max_eval,log_name="cma_log_{}_{}.csv".format(i,k),log_path=log_path)
            # target step-size for convergence
            target_std = 1e-6
            # target_std = 1e-20

            """
            set optimizer
            """
            init_m, init_sigma = np.random.rand(d) * 2 - 1, 2
            lam = cma.CMAParam.pop_size(d)
            w_func = weight.CMAWeight(lam, min_problem=f.minimization_problem)
            _optimizer = cma._CMAES(d, w_func, m=init_m, sigma=init_sigma)
            _optimizer.set_terminate_condition(target_std)

            optimizer = copy.deepcopy(_optimizer)
            cma_f_param.set_optimizer(optimizer)
            f._evaluation(cma_f_param(np.array(target_param[k,:])),np.array([target_param[k,:]]))

    # print('suggest')
    # print('evals:{}'.format(np.mean(experiment_evals_suggest,axis=1)))
    # print('argEval:{}'.format(np.mean(experiment_evals_suggest)))
    # print('cmaes')
    # print('evals:{}'.format(np.mean(experiment_evals_cmaes,axis=1)))
    # print('argEval:{}'.format(np.mean(experiment_evals_cmaes)))
    # print('wscmaes')
    # print('evals:{}'.format(np.mean(experiment_evals_wscmaes,axis=1)))
    # print('argEval:{}'.format(np.mean(experiment_evals_wscmaes)))

    return d

if __name__ == '__main__':
    # 実験回数
    experiment_times = 4
    dimention = 2  #仮
    target_param_num = 1

    # ログ設定
    log_path = os.path.dirname(__file__) + "/log/"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    # 各評価回数格納用
    c_gausian_cma_evals = np.zeros(experiment_times*target_param_num)
    cma_evals = np.zeros(experiment_times*target_param_num)
    wscma_evals = np.zeros(experiment_times*target_param_num)
    
    # 実験開始
    dimention = experiment_evaluate(experiment_times=experiment_times,log_path=log_path)

# bench = bench.FechPush()
# X = np.array([[0,0,0],[0,0,0]])
# # param = np.array([[0.05,0,0,0],[0,0,0.05,0]])
# param = np.array([0.05,0,0,0])
# print(bench._evaluation(X,param,video_pass="./src/experiment_simulate/video"))