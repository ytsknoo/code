from pickletools import optimize
import numpy as np
import pandas as pd
import os, sys
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib

import copy
import pickle


from GPy.kern import Kern, RBF, Bias, Linear, Coregionalize, Matern52, Matern32
from GPy.util.multioutput import LCM, ICM
from GPyOpt import Design_space
from GPyOpt.experiment_design.random_design import RandomDesign

import math


if __name__ == '__main__':
    move = ['Linear Shift','Nonlinear Shift','Noisy Shift']
    # label = ['Sphere(Nonlinear Shift) 10','Sphere(Nonlinear Shift) 15','Sphere(Nonlinear Shift) 20']

    fig,ax = plt.subplots(nrows=1, ncols=3,sharey="all",figsize=(12,3.5))
    plt.subplots_adjust(wspace=0.1)
    shareynotall = False
    fname = "Easom"
    
    for i in range(3):
        log = pd.read_csv(os.path.dirname(__file__)+"/plot_log_{}_easom.csv".format(i))

        log = log.iloc[:,1:]

        suggestfcalls = log.iloc[:,0].dropna(how='all')
        suggestevals = log.iloc[:,1].dropna(how='all')
        suggestevals75 = log.iloc[:,2].dropna(how='all')
        suggestevals25 = log.iloc[:,3].dropna(how='all')
        cmafcalls = log.iloc[:,4].dropna(how='all')
        cmaevals = log.iloc[:,5].dropna(how='all')
        cmaevals75 = log.iloc[:,6].dropna(how='all')
        cmaevals25 = log.iloc[:,7].dropna(how='all')
        wscmafcalls = log.iloc[:,8].dropna(how='all')
        wscmaevals = log.iloc[:,9].dropna(how='all')
        wscmaevals75 = log.iloc[:,10].dropna(how='all')
        wscmaevals25 = log.iloc[:,11].dropna(how='all')
        concmafcalls = log.iloc[:,12].dropna(how='all')
        concmaevals = log.iloc[:,13].dropna(how='all')
        concmaevals25 = log.iloc[:,14].dropna(how='all')
        concmaevals75 = log.iloc[:,15].dropna(how='all')
        suggestmeanevals = log.iloc[:,16].dropna(how='all')
        suggestmeanevals25 = log.iloc[:,17].dropna(how='all')
        suggestmeanevals75 = log.iloc[:,18].dropna(how='all')


        ttl = fname + "({})".format(move[i])
        # ttl = label[i]
  
        marker_num = 9

        sugest_mark_point = np.arange(0,len(suggestfcalls),int(len(concmafcalls)/marker_num))
        cma_mark_point = np.arange(0,len(cmafcalls),int(len(concmafcalls)/marker_num))
        wscma_mark_point = np.arange(0,len(wscmafcalls),int(len(concmafcalls)/marker_num))
        concma_mark_point = np.arange(0,len(concmafcalls),int(len(concmafcalls)/marker_num))

        # ax = fig.add_subplot(1, 3, i+1)
        
        p3 = ax[i].plot(suggestfcalls,suggestevals,"-",marker="o",markevery=sugest_mark_point,color="#1f77b4")
        p7 = ax[i].plot(wscmafcalls,wscmaevals,marker="D",markevery=wscma_mark_point,color='#9467bd')
        # p4 = ax[i].plot(cmafcalls,cmaevals,marker="^",markevery=cma_mark_point,color='#949494')
        p4 = ax[i].plot(cmafcalls,cmaevals,marker="^",markevery=cma_mark_point,color='#ff7f0e')
        p5 = ax[i].plot(concmafcalls,concmaevals,"--",marker="s",markevery=concma_mark_point,color="#2ca02c")
        p6 = ax[i].plot(concmafcalls,suggestmeanevals,"--",marker="h",markevery=concma_mark_point,alpha=0.5,color="#1f77b4")

        ax[i].fill_between(suggestfcalls, suggestevals75, suggestevals25, alpha=0.2, color='#1f77b4')
        ax[i].fill_between(wscmafcalls, wscmaevals75, wscmaevals25, alpha=0.2, color='#9467bd')
        ax[i].fill_between(cmafcalls, cmaevals75, cmaevals25, alpha=0.2, color='#ff7f0e')
        # ax[i].fill_between(cmafcalls, cmaevals75, cmaevals25, alpha=0.2, color='#949494')
        ax[i].fill_between(concmafcalls, concmaevals75, concmaevals25, alpha=0.2, color='#2ca02c')
        ax[i].fill_between(concmafcalls, suggestmeanevals75, suggestmeanevals25, alpha=0.1, color='#1f77b4')
        



        ax[i].set_title(ttl,fontsize=15,pad=10)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        #fontsizeでx軸の文字サイズ変更
        ax[i].set_xlabel("x_label",fontsize=13)
        #fontsizeでy軸の文字サイズ変更
        ax[i].set_ylabel("y_label",fontsize=13)

        if i == 1 :
            # ax[i].legend((p3[0],p6[0],p4[0],p5[0],p7[0]), ("Proposed","Proposed (model output)","CMA-ES","Contextual CMA-ES","WS-CMA-ES"),ncol=5,loc='upper center', bbox_to_anchor=(.5, -.2))
            # ax[i].legend((p3[0],p4[0],p5[0],p7[0]), ("Proposed","CMA-ES","Contextual CMA-ES","WS-CMA-ES"),ncol=2,loc='upper center', bbox_to_anchor=(.5, -.2))
            ax[i].set_xlabel('Num. of Evaluations',labelpad=10)
        else:
            ax[i].set_xlabel('')

        
        if i == 0: 
            ax[i].set_ylabel('Best Evaluation Value',labelpad=10)
        else:
            ax[i].set_ylabel('')

        


        # log変換
        ax[i].set_yscale('log')

        ax[i].grid()

    if shareynotall:
        ymin, ymax = ax[1].get_ylim()
        ax[2].set_ylim(ymin,ymax)
    
    y_min, y_max = ax[0].get_ylim()
    ax[0].set_ylim(5e-9, y_max)


    plt.savefig(os.path.dirname(__file__) + "/median_best_eval.pdf",bbox_inches='tight')
    plt.savefig(os.path.dirname(__file__) + "/median_best_eval.png",bbox_inches='tight')