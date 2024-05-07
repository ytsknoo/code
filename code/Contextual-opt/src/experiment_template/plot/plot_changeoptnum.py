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
    label = ['Sphere(Nonlinear Shift) 10','Sphere(Nonlinear Shift) 15','Sphere(Nonlinear Shift) 20']

    fig,ax = plt.subplots(nrows=1, ncols=3,sharey="all",figsize=(12,3.5))
    plt.subplots_adjust(wspace=0.1)
    shareynotall = False
    fname = "Sphere"
    
    for i in range(3):
        log05 = pd.read_csv(os.path.dirname(__file__)+"/plot_log_{}_sphere05.csv".format(i))
        log10 = pd.read_csv(os.path.dirname(__file__)+"/plot_log_{}_sphere10.csv".format(i))
        log15 = pd.read_csv(os.path.dirname(__file__)+"/plot_log_{}_sphere15.csv".format(i))
        log20 = pd.read_csv(os.path.dirname(__file__)+"/plot_log_{}_sphere20.csv".format(i))

        log05 = log05.iloc[:,1:]
        log10 = log10.iloc[:,1:]
        log15 = log15.iloc[:,1:]
        log20 = log20.iloc[:,1:]

        suggestfcalls05 = log05.iloc[:,0].dropna(how='all')
        suggestevals05 = log05.iloc[:,1].dropna(how='all')
        suggestevals7505 = log05.iloc[:,2].dropna(how='all')
        suggestevals2505 = log05.iloc[:,3].dropna(how='all')

        suggestfcalls10 = log10.iloc[:,0].dropna(how='all')
        suggestevals10 = log10.iloc[:,1].dropna(how='all')
        suggestevals7510 = log10.iloc[:,2].dropna(how='all')
        suggestevals2510 = log10.iloc[:,3].dropna(how='all')

        cmafcalls10 = log10.iloc[:,4].dropna(how='all')
        cmaevals10 = log10.iloc[:,5].dropna(how='all')
        cmaevals7510 = log10.iloc[:,6].dropna(how='all')
        cmaevals2510 = log10.iloc[:,7].dropna(how='all')

        suggestfcalls15 = log15.iloc[:,0].dropna(how='all')
        suggestevals15 = log15.iloc[:,1].dropna(how='all')
        suggestevals7515 = log15.iloc[:,2].dropna(how='all')
        suggestevals2515 = log15.iloc[:,3].dropna(how='all')

        suggestfcalls20 = log20.iloc[:,0].dropna(how='all')
        suggestevals20 = log20.iloc[:,1].dropna(how='all')
        suggestevals7520 = log20.iloc[:,2].dropna(how='all')
        suggestevals2520 = log20.iloc[:,3].dropna(how='all')


        ttl = fname + "({})".format(move[i])
        # ttl = label[i]
  
        marker_num = 9

        sugest_mark_point05 = np.arange(0,len(suggestfcalls05),int(len(cmafcalls10)/marker_num))
        sugest_mark_point10 = np.arange(0,len(suggestfcalls10),int(len(cmafcalls10)/marker_num))
        sugest_mark_point15 = np.arange(0,len(suggestfcalls15),int(len(cmafcalls10)/marker_num))
        sugest_mark_point20 = np.arange(0,len(suggestfcalls20),int(len(cmafcalls10)/marker_num))
        cma_mark_point10 = np.arange(0,len(cmafcalls10),int(len(cmafcalls10)/marker_num))

        # ax = fig.add_subplot(1, 3, i+1)
        
        p1 = ax[i].plot(suggestfcalls05,suggestevals05,"-",marker="D",markevery=sugest_mark_point05,color="#bcbd22")
        p2 = ax[i].plot(suggestfcalls10,suggestevals10,"-",marker="o",markevery=sugest_mark_point10,color="#1f77b4")
        p3 = ax[i].plot(suggestfcalls15,suggestevals15,"-",marker="s",markevery=sugest_mark_point15,color="#e377c2")
        p4 = ax[i].plot(suggestfcalls20,suggestevals20,"-",marker="p",markevery=sugest_mark_point20,color="#d62728")
        p5 = ax[i].plot(cmafcalls10,cmaevals10,"-",marker="^",markevery=cma_mark_point10,color="#ff7f0e")

        ax[i].fill_between(suggestfcalls05, suggestevals7505, suggestevals2505, alpha=0.2, color='#bcbd22')
        ax[i].fill_between(suggestfcalls10, suggestevals7510, suggestevals2510, alpha=0.2, color='#1f77b4')
        ax[i].fill_between(suggestfcalls15, suggestevals7515, suggestevals2515, alpha=0.2, color='#e377c2')
        ax[i].fill_between(suggestfcalls20, suggestevals7520, suggestevals2520, alpha=0.2, color='#d62728')
        ax[i].fill_between(cmafcalls10, cmaevals7510, cmaevals2510, alpha=0.2, color='#ff7f0e')



        ax[i].set_title(ttl,fontsize=15,pad=10)

        matplotlib.rcParams['pdf.fonttype'] = 42
        matplotlib.rcParams['ps.fonttype'] = 42
        #fontsizeでx軸の文字サイズ変更
        ax[i].set_xlabel("x_label",fontsize=13)
        #fontsizeでy軸の文字サイズ変更
        ax[i].set_ylabel("y_label",fontsize=13)

        if i == 1 :
            ax[i].legend((p1[0],p2[0],p3[0],p4[0],p5[0]), (r"$M_{prev}=5$",r"$M_{prev}=10$",r"$M_{prev}=15$",r"$M_{prev}=20$","CMA-ES"),ncol=5,loc='upper center', bbox_to_anchor=(.5, -.2))
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


    plt.savefig(os.path.dirname(__file__) + "/median_best_eval.pdf",bbox_inches='tight')
    plt.savefig(os.path.dirname(__file__) + "/median_best_eval.png",bbox_inches='tight')