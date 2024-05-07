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

    fig,ax = plt.subplots(nrows=1, ncols=2,sharey="all",figsize=(10,3.5))
    plt.subplots_adjust(wspace=0.07)
    shareynotall = False
    # fname = ["FetchPush-v2","FetchSlide-v2"]
    fname = ["point designation (4 dim.)","angle designation (3 dim.)"]
    # sp = ["push","slide"]
    sp = [4,3]
    
    for i in range(2):
        # log = pd.read_csv(os.path.dirname(__file__)+"/plot_log_{}_3dim.csv".format(sp[i]))
        log = pd.read_csv(os.path.dirname(__file__)+"/plot_log_push_{}dim_pena.csv".format(sp[i]))

        log = log.iloc[:,1:]

        concmafcalls = log.iloc[:,12].dropna(how='all')
        suggestfcalls = log.iloc[:,19].dropna(how='all')
        suggestevals = log.iloc[:,20].dropna(how='all')
        suggestevals75 = log.iloc[:,21].dropna(how='all')
        suggestevals25 = log.iloc[:,22].dropna(how='all')
        cmafcalls = log.iloc[:,23].dropna(how='all')
        cmaevals = log.iloc[:,24].dropna(how='all')
        cmaevals75 = log.iloc[:,25].dropna(how='all')
        cmaevals25 = log.iloc[:,26].dropna(how='all')
        wscmafcalls = log.iloc[:,27].dropna(how='all')
        wscmaevals = log.iloc[:,28].dropna(how='all')
        wscmaevals75 = log.iloc[:,29].dropna(how='all')
        wscmaevals25 = log.iloc[:,30].dropna(how='all')


        ttl = fname [i]
        # ttl = label[i]
  
        marker_num = 9

        sugest_mark_point = np.arange(0,len(suggestfcalls),int(len(concmafcalls)/marker_num))
        cma_mark_point = np.arange(0,len(cmafcalls),int(len(concmafcalls)/marker_num))
        wscma_mark_point = np.arange(0,len(wscmafcalls),int(len(concmafcalls)/marker_num))
        concma_mark_point = np.arange(0,len(concmafcalls),int(len(concmafcalls)/marker_num))

        # ax = fig.add_subplot(1, 3, i+1)
        
        p3 = ax[i].plot(suggestfcalls,suggestevals,"-",marker="o",markevery=sugest_mark_point,color="#1f77b4")
        p7 = ax[i].plot(wscmafcalls,wscmaevals,marker="D",markevery=wscma_mark_point,color='#9467bd')
        p4 = ax[i].plot(cmafcalls,cmaevals,marker="^",markevery=cma_mark_point,color='#ff7f0e')
  

        ax[i].fill_between(suggestfcalls, suggestevals75, suggestevals25, alpha=0.2, color='#1f77b4')
        ax[i].fill_between(cmafcalls, cmaevals75, cmaevals25, alpha=0.2, color='#ff7f0e')
        ax[i].fill_between(wscmafcalls, wscmaevals75, wscmaevals25, alpha=0.2, color='#9467bd')
        



        ax[i].set_title(ttl,fontsize=15,pad=10)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        #fontsizeでx軸の文字サイズ変更
        ax[i].set_xlabel("x_label",fontsize=13)
        #fontsizeでy軸の文字サイズ変更
        ax[i].set_ylabel("y_label",fontsize=13)

        # ax[i].legend((p3[0],p6[0],p4[0],p5[0],p7[0]), ("Proposed","Proposed (model output)","CMA-ES","Contextual CMA-ES","WS-CMA-ES"),ncol=5,loc='upper center', bbox_to_anchor=(.5, -.2))
        # ax[i].set_xlabel('Num. of Evaluations',labelpad=10)

        # ax[i].set_ylabel('Best Evaluation Value',labelpad=10)

        if i == 0:
            ax[i].legend((p3[0],p4[0],p7[0]), ("Proposed","CMA-ES","WS-CMA-ES"),ncol=5,loc='upper center', bbox_to_anchor=(1.03, -.2))
            ax[i].set_xlabel('Num. of Evaluations',labelpad=10,x=1)
        else:
            ax[i].set_xlabel('')

        
        if i == 0: 
            ax[i].set_ylabel('Penalty',labelpad=10)
        else:
            ax[i].set_ylabel('')


        


        # log変換
        # ax[i].set_yscale('log')

        ax[i].grid()

        ax[i].set_xlim(0,500)
        # ax[i].set_ylim(2e-4,1)

    # if shareynotall:
    #     ymin, ymax = ax[1].get_ylim()
    #     ax[2].set_ylim(ymin,ymax)
    
    # y_min, y_max = ax[0].get_ylim()
    # ax[0].set_ylim(5e-9, y_max)


    plt.savefig(os.path.dirname(__file__) + "/median_penalty.pdf",bbox_inches='tight')
    plt.savefig(os.path.dirname(__file__) + "/median_penalty.png",bbox_inches='tight')