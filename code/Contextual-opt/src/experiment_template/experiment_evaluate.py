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
from src.objective_function import benchmark as bench
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

"""
experiment
"""
def experiment_evaluate(experiment_times=1,log_path=None,verpose=True,verbose=True):
    d = 2
    max_eval = 4e4
    f = bench.ShiftSphere(d)
    # f = bench.NonLinearShiftSphere(d)
    # f = bench.NoisedShiftSphere(d)

    # f = bench.ShiftEasom(2)
    # f = bench.NonLinearShiftEasom(2)
    # f = bench.NoisedShiftEasom(2)

    # f = bench.ShiftRosenbrock(d)
    # f = bench.NonLinearRosenbrock(d)
    # f = bench.NoisedRosenbrock(d)

    # f = bench.ShiftEllipsoid(d)
    # f = bench.RotateEllipsoid(d)
    # f = bench.ShiftRastrigin(d)
    # f = bench.NonLinearShiftRastrigin(d)
    # f = bench.NoisedShiftRastrigin(d)
    f_param = ParameterFunction(f, max_eval=max_eval)
    init_sample_num = 10
    target_param_num = 1

    """
    Evaluate trained model
    """

  

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
    np.random.seed(20)
    for i in range(experiment_times):

        # 初期サンプル生成
        init = InitSample(d,f,init_sample_num,max_eval)

        init.init_optimize()
        init_param = init.get_init_param()
        best_solutions = init.get_best_solutions()
        delete_list = np.array([])

        # 最適化失敗してたら削除
        # for l in range(len(best_solutions)):
        #     if f._evaluation([best_solutions[l]],init_param[l]) > 1e-07:
        #         delete_list = np.append(delete_list,l)
        
        # for m in range(len(delete_list)):
        #     best_solutions = np.delete(best_solutions,int(delete_list[m]),0)
        #     init_param = np.delete(init_param,int(delete_list[m]),0)
        
        # delete_list = np.array([])
                


        # ターゲットパラメータ決定
        param_range = np.array(f.param_range)
        for j in range(target_param_num):
            target_param[j,:] = np.random.random((1,len(param_range)))*(param_range[:,1]-param_range[:,0]).T+param_range[:,0].T


        ConGauCma = ContextualGausianCmaesAcc(d,f,init_param,best_solutions,log_path=log_path)
        ConGauCma.create_model()
        WSCmaes = WarmStartCmaes(d,f,init_param,best_solutions,max_eval,log_path=log_path)
        test = np.array([target_param[0,:]])

        for k in range(target_param_num):
            experiment_evals_suggest[i,k] = ConGauCma.optimize(np.array([target_param[k,:]]),logname="c_gausian_cmaes_log_{}_{}.csv".format(i,k))
            experiment_evals_wscmaes[i,k] = WSCmaes.optimize(np.array([target_param[k,:]]),logname="ws_cmaes_log_{}_{}.csv".format(i,k))

            # CMA-ES
            cma_f_param = ParameterFunction(f, max_eval,log_name="cma_log_{}_{}.csv".format(i,k),log_path=log_path)
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

            cma_f_param.set_optimizer(_optimizer)
            experiment_evals_cmaes[i,k] = f._evaluation(cma_f_param(np.array([target_param[k,:]])),np.array([target_param[k,:]]))

    print('suggest')
    print('evals:{}'.format(np.mean(experiment_evals_suggest,axis=1)))
    print('argEval:{}'.format(np.mean(experiment_evals_suggest)))
    print('cmaes')
    print('evals:{}'.format(np.mean(experiment_evals_cmaes,axis=1)))
    print('argEval:{}'.format(np.mean(experiment_evals_cmaes)))
    print('wscmaes')
    print('evals:{}'.format(np.mean(experiment_evals_wscmaes,axis=1)))
    print('argEval:{}'.format(np.mean(experiment_evals_wscmaes)))

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

    n = 0
    sugestbestevals = pd.DataFrame()
    cmabestevals = pd.DataFrame()
    wscmabestevals = pd.DataFrame()

    sugestmeanvecevals = pd.DataFrame()

    sugestfcalls = pd.DataFrame()
    cmafcalls = pd.DataFrame()
    wscmafcalls = pd.DataFrame()

    suggestplot_len = np.array([])
    cmaplot_len = np.array([])
    wscmaplot_len = np.array([])

    # #グラフ描画
    for i in range(experiment_times):
        for j in range(target_param_num):
            df = pd.read_csv(log_path + '/c_gausian_cmaes_log_{}_{}.csv'.format(i,j))
            df2 = pd.read_csv(log_path + '/cma_log_{}_{}.csv'.format(i,j))
            df3 = pd.read_csv(log_path + '/ws_cmaes_log_{}_{}.csv'.format(i,j))
            if len(sugestfcalls.index) < len(df.index):
                sugestfcalls = df.loc[:,"EvalCount"]
            if len(cmafcalls.index) < len(df2.index):
                cmafcalls = df2.loc[:,"EvalCount"]
            if len(wscmafcalls.index) < len(df3.index):
                wscmafcalls = df3.loc[:,"EvalCount"]
            
            # 評価回数保存
            suggestplot_len = np.append(suggestplot_len,len(df.index))
            cmaplot_len = np.append(cmaplot_len,len(df2.index))
            wscmaplot_len = np.append(wscmaplot_len,len(df3.index))

            sugestbestevals = pd.concat([sugestbestevals,df.loc[:,"BestEval"]], axis=1)
            cmabestevals = pd.concat([cmabestevals,df2.loc[:,"BestEval"]], axis=1)
            wscmabestevals = pd.concat([wscmabestevals,df3.loc[:,"BestEval"]], axis=1)

            sugestmeanvecevals = pd.concat([sugestmeanvecevals,df.loc[:,"MeanEval"]], axis=1)

            c_gausian_cma_evals[n] = df.iloc[-1,1]
            cma_evals[n] = df2.iloc[-1,1]
            wscma_evals[n] = df3.iloc[-1,1]

            n += 1
    
    #最良解で埋める
    sugestbestevals = sugestbestevals.fillna(method='ffill')
    cmabestevals = cmabestevals.fillna(method='ffill')
    wscmabestevals = wscmabestevals.fillna(method='ffill')

    #log変換
    # sugestbestevals_log = np.log10(sugestbestevals)
    # cmabestevals_log = np.log10(cmabestevals)

    # 中央値，四分位数を計算
    sugestbestevals_median = sugestbestevals.median(axis=1)
    cmabestevals_median = cmabestevals.median(axis=1)
    wscmabestevals_median = wscmabestevals.median(axis=1)
    sugestmeanvecevals_median = sugestmeanvecevals.median(axis=1)
    sugestbestevals_quan25 = sugestbestevals.quantile(0.25,axis=1)
    cmabestevals_quan25 = cmabestevals.quantile(0.25,axis=1)
    wscmabestevals_quan25 = wscmabestevals.quantile(0.25,axis=1)
    sugestmeanvecevals_quan25 = sugestmeanvecevals.quantile(0.25,axis=1)
    sugestbestevals_quan75 = sugestbestevals.quantile(0.75,axis=1)
    cmabestevals_quan75 = cmabestevals.quantile(0.75,axis=1)
    wscmabestevals_quan75 = wscmabestevals.quantile(0.75,axis=1)
    sugestmeanvecevals_quan75 = sugestmeanvecevals.quantile(0.75,axis=1)

    # プロット範囲計算
    
    suggestplot_len = int(np.median(suggestplot_len))
    cmaplot_len = int(np.median(cmaplot_len))
    wscmaplot_len = int(np.median(wscmaplot_len))

    # ContextualCmaesの値
    # if suggestplot_len > cmaplot_len :
    #     concmaplot_len = suggestplot_len
    # else:
    #     concmaplot_len = cmaplot_len

    concmaplot_len = max([suggestplot_len,cmaplot_len,wscmaplot_len])
    
    if max([len(sugestfcalls),len(cmafcalls),len(wscmafcalls)]) == len(sugestfcalls) :
        concmafcalls = sugestfcalls
    elif max([len(sugestfcalls),len(cmafcalls),len(wscmafcalls)]) == len(cmafcalls) :
        concmafcalls = cmafcalls
    elif max([len(sugestfcalls),len(cmafcalls),len(wscmafcalls)]) == len(wscmafcalls) :
        concmafcalls = wscmafcalls

    # CCMA-ESの中央値，第一四分位数，第三四分位数
    # Shift
    # linear
    # Concmaeval = 2.8801e-16
    # Concmaeval25 = 2.4915e-16
    # Concmaeval75 = 4.7827e-16 

    # nonlinear
    # Concmaeval = 85.778
    # Concmaeval25 = 52.0599
    # Concmaeval75 = 137.7584

    # noisy
    # Concmaeval = 83.6523 
    # Concmaeval25 = 60.6052 
    # Concmaeval75 = 104.3701

    # Easom
    # linear
    # Concmaeval = 0
    # Concmaeval25 = 0
    # Concmaeval75 = 4.996e-16

    # nonlinear
    # Concmaeval = 0.99998 
    # Concmaeval25 = 0.80814
    # Concmaeval75 = 1

    # noisy
    # Concmaeval = 1
    # Concmaeval25 = 0.94529
    # Concmaeval75 = 1

    # Rosenbrock
    # linear
    # Concmaeval = 1.2338e-13
    # Concmaeval25 = 6.2549e-14
    # Concmaeval75 = 3.9144e-13

    # nonlinear
    # Concmaeval = 55326.7342
    # Concmaeval25 = 36706.6587
    # Concmaeval75 = 243892.4248

    # noisy
    Concmaeval = 85921.4696
    Concmaeval25 = 48004.7496
    Concmaeval75 = 157621.3328


    # plt.title(fname,fontsize=15)
    # Concmaeval = np.log10(Concmaeval)

    Concmaevals = np.full(concmaplot_len,Concmaeval)
    Concmaevals = Concmaevals.reshape(concmaplot_len,1)
    Concmaevals = pd.DataFrame(Concmaevals)
    Concmaevals25 = np.full(concmaplot_len,Concmaeval25)
    Concmaevals25 = Concmaevals25.reshape(concmaplot_len,1)
    Concmaevals25 = pd.DataFrame(Concmaevals25)
    Concmaevals75 = np.full(concmaplot_len,Concmaeval75)
    Concmaevals75 = Concmaevals75.reshape(concmaplot_len,1)
    Concmaevals75 = pd.DataFrame(Concmaevals75)


    sugestmeanvecevals_plot = np.full(concmaplot_len,sugestmeanvecevals_median[0])
    sugestmeanvecevals_plot = sugestmeanvecevals_plot.reshape(concmaplot_len,1)
    sugestmeanvecevals_plot = pd.DataFrame(sugestmeanvecevals_plot)
    sugestmeanvecevals_plot25 = np.full(concmaplot_len,sugestmeanvecevals_quan25[0])
    sugestmeanvecevals_plot25 = sugestmeanvecevals_plot25.reshape(concmaplot_len,1)
    sugestmeanvecevals_plot25 = pd.DataFrame(sugestmeanvecevals_plot25)
    sugestmeanvecevals_plot75 = np.full(concmaplot_len,sugestmeanvecevals_quan75[0])
    sugestmeanvecevals_plot75 = sugestmeanvecevals_plot75.reshape(concmaplot_len,1)
    sugestmeanvecevals_plot75 = pd.DataFrame(sugestmeanvecevals_plot75)

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    #fontsizeでx軸の文字サイズ変更
    plt.xlabel("x_label",fontsize=13)
    #fontsizeでy軸の文字サイズ変更
    plt.ylabel("y_label",fontsize=13)

    plt.tight_layout()

    marker_num = 9

    concma_mark_point = np.arange(0,concmaplot_len,int(concmaplot_len/marker_num))
    sugest_mark_point = np.arange(0,suggestplot_len,int(concmaplot_len/marker_num))
    cma_mark_point = np.arange(0,cmaplot_len,int(concmaplot_len/marker_num))
    wscma_mark_point = np.arange(0,wscmaplot_len,int(concmaplot_len/marker_num))
    
    p3 = plt.plot(sugestfcalls[:suggestplot_len],sugestbestevals_median[:suggestplot_len],"-",marker="o",markevery=sugest_mark_point,color="#1f77b4")
    p4 = plt.plot(cmafcalls[:cmaplot_len],cmabestevals_median[:cmaplot_len],marker="^",markevery=cma_mark_point,color='#ff7f0e')
    p5 = plt.plot(concmafcalls[:concmaplot_len],Concmaevals,"--",marker="s",markevery=concma_mark_point,color="#2ca02c")
    p6 = plt.plot(concmafcalls[:concmaplot_len],sugestmeanvecevals_plot,"--",marker="s",markevery=concma_mark_point,alpha=0.5,color="#1f77b4")
    p7 = plt.plot(wscmafcalls[:wscmaplot_len],wscmabestevals_median[:wscmaplot_len],marker="^",markevery=wscma_mark_point,color='#d62728')

    plt.fill_between(sugestfcalls[:suggestplot_len], sugestbestevals_quan75[:suggestplot_len], sugestbestevals_quan25[:suggestplot_len], alpha=0.2, color='#1f77b4')
    plt.fill_between(cmafcalls[:cmaplot_len], cmabestevals_quan75[:cmaplot_len], cmabestevals_quan25[:cmaplot_len], alpha=0.2, color='#ff7f0e')
    plt.fill_between(wscmafcalls[:wscmaplot_len], wscmabestevals_quan75[:wscmaplot_len], wscmabestevals_quan25[:wscmaplot_len], alpha=0.2, color='#d62728')


    

    # plt.legend((p4[0],p7[0]), ("CMA-ES","WS-CMA-ES"))
    plt.legend((p3[0],p4[0],p5[0],p6[0]), ("Proposed","CMA-ES","Contextual CMA-ES","Proposed pred meaneval"))
    plt.xlabel('Num. of Evaluations')
    plt.ylabel('Best Evaluation Value')

    # log変換
    plt.yscale('log')

    plt.grid()

    plt.savefig(log_path + "/median_best_eval.pdf")
    plt.savefig(log_path + "/median_best_eval.png")

    # 記録用に保存
    plot_log = pd.concat([sugestfcalls[:suggestplot_len],sugestbestevals_median[:suggestplot_len],sugestbestevals_quan75[:suggestplot_len],sugestbestevals_quan25[:suggestplot_len],cmafcalls[:cmaplot_len],cmabestevals_median[:cmaplot_len],cmabestevals_quan75[:cmaplot_len],cmabestevals_quan25[:cmaplot_len],wscmafcalls[:wscmaplot_len],wscmabestevals_median[:wscmaplot_len],wscmabestevals_quan75[:wscmaplot_len],wscmabestevals_quan25[:wscmaplot_len],concmafcalls[:concmaplot_len],Concmaevals,Concmaevals25,Concmaevals75,sugestmeanvecevals_plot,sugestmeanvecevals_plot25,sugestmeanvecevals_plot75],axis=1)
    plot_log.to_csv(os.path.dirname(__file__)+"/plot/plot_log.csv")

    plt.clf()
    

    
    print("suggest")
    print("MeanFCall:{}".format(np.average(c_gausian_cma_evals)))
    print("cmaes")
    print("MeanFCall:{}".format(np.average(cma_evals)))
    print("wscmaes")
    print("MeanFCall:{}".format(np.average(wscma_evals)))