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
# from src.objective_function import fech_push as bench
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

np.random.seed(12)

class BaseObjective(object):
    def __init__(self, xd, cd):
        self.xd = xd    # 設計変数の次元数
        self.cd = cd    # 文脈ベクトルの次元数
        
        self.eval_count = 0
        
    def __call__(self, X, context):
        self.eval_count += len(X)
        return self.evaluate(X, context)
    
    def evaluate(self, X, S):
        # X: 設計変数
        # G: 文脈ベクトル
        return
    
    def generate_context(self, lam):
        # lam: サンプルサイズ
        return
    
    def terminate_condition(self):
        return False

class SphereLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        return ((X - np.dot(S, self.G)) ** 2).sum(axis=1)
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval

class SphereNonLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        X = X - np.dot(S**2, self.G)
        return (X ** 2).sum(axis=1)
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval

class SphereNoisyLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        randstate = np.random.get_state()
        np.random.seed(int(abs(np.sum(S))*10e4))
        X = X - np.dot(S, self.G) + np.random.randn(X.shape[0],X.shape[1])*0.25
        np.random.set_state(randstate)
        return (X ** 2).sum(axis=1)
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval

class RosenbrockLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        X = X - np.dot(S, self.G)
        return np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + (X[:, :-1] - 1.)**2, axis=1)
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval

class RosenbrockNonLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        X = X - np.dot(S**2, self.G)
        return np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + (X[:, :-1] - 1.)**2, axis=1)
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval

class RosenbrockNoisyLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        randstate = np.random.get_state()
        np.random.seed(int(abs(np.sum(S))*10e4))
        X = X - np.dot(S, self.G) + np.random.randn(X.shape[0],X.shape[1])*0.25
        np.random.set_state(randstate)
        return np.sum(100 * (X[:, :-1]**2 - X[:, 1:])**2 + (X[:, :-1] - 1.)**2, axis=1)
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval

class EasomLinear(BaseObjective):
    def __init__(self, xd, cd, maxeval=10000):
        super().__init__(xd, cd)
        self.maxeval = maxeval
        self.G = np.random.randn(self.cd, self.xd)
        
    def evaluate(self, X, S):
        X = X - np.dot(S, self.G)
        evals = np.zeros(len(X),)
        for i in range(len(X)):
            evals[i] = -np.cos(X[i][0])*np.cos(X[i][1])*np.exp(-X[i][1]**2 + 2*np.pi*X[i][1] - X[i][0]**2 + 2*np.pi*X[i][0] - 2*np.pi**2) + 1
        return evals
        
    def generate_context(self, lam):
        return np.random.rand(lam, self.cd) * 4 - 2
    
    def terminate_condition(self):
        return self.eval_count > self.maxeval


class ContextualCMAES(object):
    def __init__(self, xd, cd):
        self.xd = xd    # 設計変数の次元数
        self.cd = cd    # 文脈ベクトルの次元数
        self.d = self.xd + self.cd
        
        # サンプル数
        self.lam = int(4 + np.floor( 3 * np.log(self.d)) * (1 + 2 * self.cd))
        self.mu = int(np.floor(self.lam / 2))
        
        # 重み
        w = np.log(self.mu + 1/2) - np.log(np.arange(1, self.lam + 1))
        w[w < 0] = 0
        self.weights = w / np.sum(w)
        self.mueff = (np.sum(w) ** 2) / np.sum(w ** 2)
        
        # ハイパーパラメータ
        self.cc = 4 / (4 + self.d)
        self.cs = (self.mueff + 2) / (self.d + self.mueff + 3)  
        self.c1 = 2 * np.min((1, self.lam / 6)) / ((self.d + 1.3) ** 2 + self.mueff) 
        self.cmu = 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.d + 2) ** 2 + self.mueff)
        self.damping = (1 + 2 * np.sqrt((self.mueff - 1) / (self.d + 1)) - 1 + self.cs) + np.log(self.cd + 1) 
        self.chiN = (self.xd ** 0.5) * (1 - 1 / (4 * self.xd) + 1 / (21 * self.xd ** 2))
        self.reg_coeff = 1e-10
        
        # 進化パス
        self.pc = np.zeros((self.xd,1)) 
        self.ps = np.zeros((self.xd,1))
        
        # 分布パラメータ
        self.C = np.eye(self.xd)
        self.sigma = 1
        self.A = np.random.randn(self.cd + 1, self.xd) 
        
        # Baseline Function
        self.qd = int(1 + 2 * self.cd + (self.cd * (self.cd - 1)) / 2)
        
        # その他
        self.ite = 0
        self.mean_eval = np.inf
        self.target_eval = np.inf
        
        self.min_eigval = 1
        self.max_eigval = 1
        
    def optimize(self, f, target_eval=1e-8):
        self.ite = 0
        self.terminate = False
        self.target_eval = target_eval
        
        while (not self.terminate) and (not f.terminate_condition()):
            self.update(f)
            print("ite: {}, EvalCount: {}, MeanEval: {}, minEig: {}, Cond: {}".format(self.ite, f.eval_count, self.mean_eval, (self.sigma ** 2) * self.min_eigval, self.max_eigval / self.min_eigval))
            
            self.ite += 1
        return self.mean_eval
    
    def compute_mean(self, S):
        if S.ndim == 1:
            S = S[None, :]
            
        _S = np.concatenate([S, np.ones((S.shape[0], 1))], axis=1)
        return np.dot(_S, self.A)
    
    def quadratic_feature(self, S):
        Sq = np.ones((S.shape[0], self.qd))
        Sq[:, 1:self.cd+1] = S
        S2 = (S[:,None,:] * S[:,:,None]).reshape(S.shape[0], -1)
        idx = np.tri(S.shape[1]).reshape(-1)
        Sq[:, self.cd+1:] = S2[:, idx == 1]
        return Sq
            
    def update(self, f):
        # 文脈ベクトルの取得
        S = f.generate_context(self.lam)
        
        # 平均ベクトルの計算
        m = self.compute_mean(S)
        m_ave_prev = self.compute_mean(np.mean(S, axis=0))
        
        # 共分散行列の平方根行列の計算
        D, B = scipy.linalg.eigh(self.C)
        sqrtC = np.dot(B, np.diag(np.sqrt(D))).dot(B.T)
        invSqrtC = np.dot(B, np.diag(np.sqrt(1 / D))).dot(B.T)
        
        # サンプル生成
        Z = np.random.randn(self.lam, self.xd)
        Y = np.dot(Z, sqrtC.T)
        X = m + self.sigma * Y
        
        # 評価
        evals = f(X, S)
        self.mean_eval = np.mean(evals)
        
        # Baseline Functionで補正した評価値の計算
        Sq = self.quadratic_feature(S)
        res = np.linalg.lstsq(Sq, evals, rcond=None)
        beta = res[0]
        ave_evals = evals - np.dot(Sq, beta)
        
        # 重みの計算
        idx = np.argsort(ave_evals)
        weights = np.zeros_like(self.weights)
        weights[idx] = self.weights
        
        # 平均ベクトルの係数の更新
        _S = np.concatenate([S, np.ones((self.lam, 1))], axis=1)
        P = (_S.T).dot(np.diag(weights)).dot(_S) + self.reg_coeff * np.eye(self.cd + 1)
        Pinv = np.linalg.inv(P)
        self.A = Pinv.dot(_S.T).dot(np.diag(weights)).dot(X)
        
        # 進化パスの更新
        m_ave_next = self.compute_mean(np.mean(S, axis=0))
        y = (m_ave_next - m_ave_prev) / self.sigma
        z = np.dot(y, invSqrtC)
        
        self.ps = (1.0 - self.cs) * self.ps + np.sqrt(self.cs * (2.0 - self.cs) * self.mueff) * z.T
        hsig = 1. if scipy.linalg.norm(self.ps) / (np.sqrt(1. - (1. - self.cs) ** (2 * (self.ite + 1)))) < (1.4 + 2. / (self.xd + 1.)) * self.chiN else 0.
        self.pc = (1. - self.cc) * self.pc + hsig * np.sqrt(self.cc * (2. - self.cc) * self.mueff) * y.T
        
        # 共分散行列の更新
        mu_update = np.dot(weights * Y.T, Y) - weights.sum() * self.C
        h = (1. - hsig) * self.c1 * self.cc * (2. - self.cc)
        self.C = self.C + h * self.C + self.c1 * (np.outer(self.pc, self.pc) - self.C) + self.cmu * mu_update
        
        # ステップサイズの更新
        self.sigma = self.sigma * np.exp(self.cs / self.damping * (scipy.linalg.norm(self.ps) / self.chiN - 1.))
        
        # 終了条件の確認
        self.min_eigval = np.min(D) * (self.sigma ** 2)
        self.max_eigval = np.max(D) * (self.sigma ** 2)
        self.terminate = self.min_eigval < 1e-30 or self.mean_eval < self.target_eval

# xd: 設計変数の次元数
# cd: 文脈ベクトルの次元数
# maxeval: 最大評価回数

experiment_times = 2

log = np.zeros(experiment_times)

for i in range(experiment_times):

    # f = SphereLinear(xd=20, cd=2, maxeval=1e5)
    f = bench.FechSlide(max_eval=5e3,log_video=False)
    # f = bench.FechSlide(max_eval=5e3,log_video=False)
    # f = SphereNonLinear(xd=20, cd=2, maxeval=1e5)
    # f = SphereNoisyLinear(xd=20, cd=2, maxeval=1e5)
    # f = RosenbrockLinear(xd=20, cd=2, maxeval=4e5)
    # f = RosenbrockNonLinear(xd=20, cd=2, maxeval=4e5)
    # f = RosenbrockNoisyLinear(xd=20, cd=2, maxeval=4e5)
    # f = EasomLinear(xd=2, cd=2, maxeval=1e5)
    # f = EasomNonLinear(xd=2, cd=2, maxeval=1e5)
    # f = EasomNoisyLinear(xd=2, cd=2, maxeval=1e5)

    opt = ContextualCMAES(f.d, f.param_dim)
    log[i] = opt.optimize(f, target_eval=1e-8)

print(log)
# print("第1四分位数：{}".format(np.percentile(log,25)))
# print("中央値：{}".format(np.median(log)))
# print("第3四分位数：{}".format(np.percentile(log,75)))