from pickletools import optimize
import numpy as np
import pandas as pd
import os, sys
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib

import copy
import pickle

# sys.path.append(os.path.dirname(__file__) + "/../../")
# sys.path.append(os.path.dirname(__file__))
# from src.objective_function.base import ParametrizedObjectiveFunction
# from src.objective_function import fech_push as bench
# from src.optimizer.base_optimizer import BaseOptimizer
# from src.optimizer import cmaes as cma
# from src.util import sampler as sampler
# from src.util import weight as weight
# from src.model.gp import bo_acq
# from src.model.gp import gp_interface as gp
# from experiment_train import ParameterFunction
# from init_sample import InitSample
# from contextual_gausian_cmaes_acc import ContextualGausianCmaesAcc
# from warm_start_cmaes import WarmStartCmaes
# 

# from GPy.kern import Kern, RBF, Bias, Linear, Coregionalize, Matern52, Matern32
# from GPy.util.multioutput import LCM, ICM
# from GPyOpt import Design_space
# from GPyOpt.experiment_design.random_design import RandomDesign

import math

if __name__ == '__main__':
    # 実験回数
    experiment_times = 20
    target_param_num = 1

    # ログ設定
    # log_path = os.path.dirname(__file__) + "/log/"
    log_path = "./src/experiment_simulate/log"
    if not os.path.isdir(log_path):
        os.mkdir(log_path)
    # 各評価回数格納用
    c_gausian_cma_evals = np.zeros(experiment_times*target_param_num)
    cma_evals = np.zeros(experiment_times*target_param_num)
    wscma_evals = np.zeros(experiment_times*target_param_num)

    n = 0
    sugestbestevals = pd.DataFrame()
    cmabestevals = pd.DataFrame()
    wscmabestevals = pd.DataFrame()

    sugestpenas = pd.DataFrame()
    cmapenas = pd.DataFrame()
    wscmapenas = pd.DataFrame()

    sugestmeanvecevals = pd.DataFrame()

    sugestfcalls = pd.DataFrame()
    cmafcalls = pd.DataFrame()
    wscmafcalls = pd.DataFrame()

    suggestplot_len = np.array([])
    cmaplot_len = np.array([])
    wscmaplot_len = np.array([])

    # 四分位範囲，中央値計算
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

            sugestpenas = pd.concat([sugestpenas,df.loc[:,"penalty"]], axis=1)
            cmapenas = pd.concat([cmapenas,df2.loc[:,"penalty"]], axis=1)
            wscmapenas = pd.concat([wscmapenas,df3.loc[:,"penalty"]], axis=1)

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

    sugestpenas_median = sugestpenas.median(axis=1)
    cmapenas_median = cmapenas.median(axis=1)
    wscmapenas_median = wscmapenas.median(axis=1)
    sugestpenas_quan25 = sugestpenas.quantile(0.25,axis=1)
    cmapenas_quan25 = cmapenas.quantile(0.25,axis=1)
    wscmapenas_quan25 = wscmapenas.quantile(0.25,axis=1)
    sugestpenas_quan75 = sugestpenas.quantile(0.75,axis=1)
    cmapenas_quan75 = cmapenas.quantile(0.75,axis=1)
    wscmapenas_quan75 = wscmapenas.quantile(0.75,axis=1)

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
    # Push
    # 設計変数4次元
    # Concmaeval = 0.111957535
    # Concmaeval25 = 0.0920272475
    # Concmaeval75 = 0.137496115
    # 設計変数3次元
    Concmaeval = 0.12950636
    Concmaeval25 = 0.11733118249999999
    Concmaeval75 = 0.139444815
    # Slide
    # 設計変数4次元
    # Concmaeval = 0.36337135
    # Concmaeval25 = 0.31058338
    # Concmaeval75 = 0.4174898975
    # 設計変数3次元
    # Concmaeval = 0.38579509
    # Concmaeval25 = 0.34904849000000004
    # Concmaeval75 = 0.4266006425


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
    
    # p3 = plt.plot(sugestfcalls[:suggestplot_len],sugestbestevals_median[:suggestplot_len],"-",marker="o",markevery=sugest_mark_point,color="#1f77b4")
    p4 = plt.plot(cmafcalls[:cmaplot_len],cmabestevals_median[:cmaplot_len],marker="^",markevery=cma_mark_point,color='#ff7f0e')
    # p5 = plt.plot(concmafcalls[:concmaplot_len],Concmaevals,"--",marker="s",markevery=concma_mark_point,color="#2ca02c")
    # p6 = plt.plot(concmafcalls[:concmaplot_len],sugestmeanvecevals_plot,"--",marker="s",markevery=concma_mark_point,alpha=0.5,color="#1f77b4")
    # p7 = plt.plot(wscmafcalls[:wscmaplot_len],wscmabestevals_median[:wscmaplot_len],marker="^",markevery=wscma_mark_point,color='#d62728')

    # plt.fill_between(sugestfcalls[:suggestplot_len], sugestbestevals_quan75[:suggestplot_len], sugestbestevals_quan25[:suggestplot_len], alpha=0.2, color='#1f77b4')
    plt.fill_between(cmafcalls[:cmaplot_len], cmabestevals_quan75[:cmaplot_len], cmabestevals_quan25[:cmaplot_len], alpha=0.2, color='#ff7f0e')
    # plt.fill_between(wscmafcalls[:wscmaplot_len], wscmabestevals_quan75[:wscmaplot_len], wscmabestevals_quan25[:wscmaplot_len], alpha=0.2, color='#d62728')


    

    # plt.legend((p4[0],p7[0]), ("CMA-ES","WS-CMA-ES"))
    # plt.legend((p3[0],p4[0],p5[0],p6[0]), ("Proposed","CMA-ES","Contextual CMA-ES","Proposed pred meaneval"))
    plt.xlabel('Num. of Evaluations')
    plt.ylabel('Best Evaluation Value')

    # log変換
    plt.yscale('log')

    plt.grid()

    plt.savefig(log_path + "/median_best_eval.pdf")
    plt.savefig(log_path + "/median_best_eval.png")

    # 記録用に保存
    plot_log = pd.concat([sugestfcalls[:suggestplot_len],sugestbestevals_median[:suggestplot_len],sugestbestevals_quan75[:suggestplot_len],sugestbestevals_quan25[:suggestplot_len],cmafcalls[:cmaplot_len],cmabestevals_median[:cmaplot_len],cmabestevals_quan75[:cmaplot_len],cmabestevals_quan25[:cmaplot_len],wscmafcalls[:wscmaplot_len],wscmabestevals_median[:wscmaplot_len],wscmabestevals_quan75[:wscmaplot_len],wscmabestevals_quan25[:wscmaplot_len],concmafcalls[:concmaplot_len],Concmaevals,Concmaevals25,Concmaevals75,sugestmeanvecevals_plot,sugestmeanvecevals_plot25,sugestmeanvecevals_plot75,sugestfcalls[:suggestplot_len],sugestpenas_median[:suggestplot_len],sugestpenas_quan75[:suggestplot_len],sugestpenas_quan25[:suggestplot_len],cmafcalls[:cmaplot_len],cmapenas_median[:cmaplot_len],cmapenas_quan75[:cmaplot_len],cmapenas_quan25[:cmaplot_len],wscmafcalls[:wscmaplot_len],wscmapenas_median[:wscmaplot_len],wscmapenas_quan75[:wscmaplot_len],wscmapenas_quan25[:wscmaplot_len]],axis=1)
    plot_log.to_csv(os.path.dirname(__file__)+"/plot_log.csv")

    plt.clf()
    

    
    print("suggest")
    print("MeanFCall:{}".format(np.average(c_gausian_cma_evals)))
    print("cmaes")
    print("MeanFCall:{}".format(np.average(cma_evals)))
    print("wscmaes")
    print("MeanFCall:{}".format(np.average(wscma_evals)))

# bench = bench.FechPush()
# X = np.array([[0,0,0],[0,0,0]])
# # param = np.array([[0.05,0,0,0],[0,0,0.05,0]])
# param = np.array([0.05,0,0,0])
# print(bench._evaluation(X,param,video_pass="./src/experiment_simulate/video"))