#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import ParametrizedObjectiveFunction
import numpy as np
import copy
import math
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# public symbols
__all__ = ['ShiftSphere', 'ShiftRosenbrock', 'ShiftEllipsoid']

class DMP(object):
    def __init__(self,goal1,goal2,scale):
        self.goal1 = np.array([goal1])
        self.goal2 = np.array([goal2])
        # self.T1 = 5
        # self.T2 = 5
        # self.T3 = 5
        self.path = np.zeros((1,len(goal1)))
        self.scale = scale



    def imitation(self):
        # 中間地点の計算
        goal05 = self.goal1/2
        goal05[0][2] = 0.145*self.scale

        # 何シミュレーション時間か決定
        self.T1 = int(np.max(np.abs(goal05))) + 1
        self.T2 = int(np.max(np.abs(self.goal1-goal05))) + 1
        self.T3 = int(np.max(np.abs(self.goal2-self.goal1))) + 1


        goal05_sig = goal05/self.T1

        for i in range(self.T1):
            self.path = np.append(self.path,goal05_sig,axis=0)

        self.path = np.delete(self.path,0,0)

        goal1_sig = (self.goal1 - goal05)/self.T2
        
        for i in range(self.T2):
            self.path = np.append(self.path,goal1_sig,axis=0)

        goal2_sig = (self.goal2 - self.goal1)/self.T3

        for i in range(self.T3):
            self.path = np.append(self.path,goal2_sig,axis=0)

        return True
    
    def get_path(self):
        return self.path

class FechPush(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, 
                    # d,                      # 次元数
                    max_eval=np.inf,        # 最大評価回数
                    param=None,             # パラメータ
                    param_range = [[-0.15,0.15],[-0.15,0.15],[-0.15,0.15],[-0.15,0.15]],  # パラメータの定義域
                    x_range = [[0,2*math.pi],[-0.2,0.2],[-0.2,0.2]],  # パラメータの定義域
                    # x_range = [[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[-0.2,0.2]],  # パラメータの定義域
                    log_video = False
                ):
        super(FechPush, self).__init__(max_eval, param, param_range)
        self.d =  3         # 問題の次元数
        # self.d =  4         # 問題の次元数
        self.param_dim = 4  # パラメータの次元数
        self.log_video = log_video
        self.x_range = x_range
        self.video_name = "CMAES"

    def set_video_name(self,str):
        self.video_name = str

    def _evaluation(self, X, param,video_pass = None):

        if not isinstance(X[0],(list,np.ndarray)):
            X = np.array([X])

        if not isinstance(param[0],(list,np.ndarray)):
            param= np.array([param])
        
        # パラメータが１つに対してX複数の時の処理
        if len(X) > len(param):
            param = np.stack([param[0] for _ in range(len(X))],axis=0)

        # 環境のインスタンスを作成
        if self.log_video:
            if (self.eval_count >= self.max_eval) and (self.max_eval % len(X) == 0):
                self.env = RecordVideo(gym.make('FetchPushDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep500")
            elif (400+len(X)/2 > self.eval_count >= 400-len(X)/2):
                self.env = RecordVideo(gym.make('FetchPushDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep400")
            elif (300+len(X)/2 > self.eval_count >= 300-len(X)/2):
                self.env = RecordVideo(gym.make('FetchPushDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep300")
            elif (200+len(X)/2 > self.eval_count >= 200-len(X)/2):
                self.env = RecordVideo(gym.make('FetchPushDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep200")
            elif (100+len(X)/2 > self.eval_count >= 100-len(X)/2):
                self.env = RecordVideo(gym.make('FetchPushDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep100")
            elif (self.eval_count == len(X)):
                self.env = RecordVideo(gym.make('FetchPushDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep1")
            else:
                self.env = gym.make('FetchPushDense-v2')
        else:
            self.env = gym.make('FetchPushDense-v2')

        #スケーリング係数
        scale = 34.5

        rewards = np.zeros(len(X))
        if not len(X) == 1:
            self.penas = np.zeros(len(X))

        for j in range(len(X)):
            # 定義域の対処
            # ペナルティの計算
            pena = 0
            for i in range(len(X[0])):
                if self.x_range[i][0] >= X[j][i] :
                    pena = pena +  np.linalg.norm(X[j][i]- self.x_range[i][0])
                elif X[j][i] >= self.x_range[i][1]:
                    pena = pena +  np.linalg.norm(X[j][i]- self.x_range[i][1])

            # 定義域でクリップ
            for i in range(len(X[0])):
                X[j][i] = np.clip(X[j][i],self.x_range[i][0],self.x_range[i][1])

            
            #オブジェクトとターゲットの位置の設定
            self.obj_pos = param[j,:2]
            self.goal_pos = param[j,2:]

            #軌道のオブジェクトからの距離
            dis_from_obj = 0.07
            goal1 = self.obj_pos

            # 軌道決定パラ
            if self.d == 3:
                goal1 = [(goal1[0] - np.sin(X[j,0])*dis_from_obj)*scale,(goal1[1] + np.cos(X[j,0])*dis_from_obj)*scale,0]
                goal2 = [X[j,1]*scale,X[j,2]*scale,0]
            elif self.d == 4:
                goal1 = [X[j,0]*scale,X[j,1]*scale,0]
                goal2 = [X[j,2]*scale,X[j,3]*scale,0]

            # 信号の取得
            dmp = DMP(goal1,goal2,scale)
            dmp.imitation()
            path = dmp.get_path()

            # 総時刻(行動回数)を指定
            T = len(path) + 5

            state = self.env.reset()
            # ターゲットとゴールの設定
            self.env.unwrapped.set_obj_pos(self.obj_pos)
            self.env.unwrapped.set_goal_pos(self.goal_pos)

            # 1エピソードのシミュレーション
            for t in range(T):
                if t < len(path):
                    action = np.append(path[t],0)
                else:
                    action = [0,0,0,0]
                
                # 状態を遷移
                next_state, reward, terminated, truncated, info = self.env.step(action)

            if not len(X) == 1:
                self.penas[j] = pena
            rewards[j] = -reward + pena
            print("{}:{}".format(self.eval_count,-reward + pena))

        self.env.close()
        return rewards
    
    def close_env(self):
        self.env.close()
    
    def get_pena(self):
        return self.penas.mean()
    
    def optimal_solution(self, param):
        # パラメータに対応する最適解（テスト用）
        # 計算ができなければNoneを返す
        return None
    
    def optimal_evaluation(self, param):
        # パラメータに対応する最良評価値（テスト用）
        # 計算ができなければNoneを返す
        return None
    
    def generate_context(self, lam):
        return np.random.rand(lam, self.param_dim) * 0.3 - 0.15

class FechSlide(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, 
                    # d,                      # 次元数
                    max_eval=np.inf,        # 最大評価回数
                    param=None,             # パラメータ
                    param_range = [[-0.1,0.1],[-0.1,0.1],[-0.3,0.3],[-0.3,0.3]],  # 文脈ベクトルの定義域
                    # x_range = [[0,math.pi],[0,0.4],[-0.4,0.4]],  # パラメータの定義域
                    x_range = [[-0.2,0],[-0.2,0.2],[0,0.4],[-0.4,0.4]],  # パラメータの定義域
                    log_video = False
                ):
        super(FechSlide, self).__init__(max_eval, param, param_range)
        # self.d =  3         # 問題の次元数
        self.d =  4         # 問題の次元数
        self.param_dim = 4  # パラメータの次元数
        self.log_video = log_video
        self.x_range = x_range
        self.video_name = "CMAES"

    def set_video_name(self,str):
        self.video_name = str

    def _evaluation(self, X, param,video_pass = None):

        if not isinstance(X[0],(list,np.ndarray)):
            X = np.array([X])

        if not isinstance(param[0],(list,np.ndarray)):
            param= np.array([param])
        
        # パラメータが１つに対してX複数の時の処理
        if len(X) > len(param):
            param = np.stack([param[0] for _ in range(len(X))],axis=0)
        

        # 環境のインスタンスを作成
        if self.log_video:
            # self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name))
            if (self.eval_count >= self.max_eval) and (self.max_eval % len(X) == 0):
                self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep500")
            elif (400+len(X)/2 > self.eval_count >= 400-len(X)/2):
                self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep400")
            elif (300+len(X)/2 > self.eval_count >= 300-len(X)/2):
                self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep300")
            elif (200+len(X)/2 > self.eval_count >= 200-len(X)/2):
                self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep200")
            elif (100+len(X)/2 > self.eval_count >= 100-len(X)/2):
                self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep == 0),name_prefix="ep100")
            elif (self.eval_count == len(X)):
                self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name),episode_trigger=(lambda ep: ep % 1 == 0),name_prefix="ep1")
            else:
                self.env = gym.make('FetchSlideDense-v2')
        else:
            self.env = gym.make('FetchSlideDense-v2')
            # self.env = gym.make('FetchSlideDense-v2',render_mode="human")
            # self.env = RecordVideo(gym.make('FetchSlideDense-v2',render_mode='rgb_array'),video_folder="./src/experiment_simulate/video/{}/".format(self.video_name))

        #スケーリング係数
        scale = 34.5

        rewards = np.zeros(len(X))
        if not len(X) == 1:
            self.penas = np.zeros(len(X))

        for j in range(len(X)):
            # 定義域の対処
            # ペナルティの計算
            pena = 0
            for i in range(len(X[0])):
                if self.x_range[i][0] >= X[j][i] :
                    pena = pena +  np.linalg.norm(X[j][i]- self.x_range[i][0])
                elif X[j][i] >= self.x_range[i][1]:
                    pena = pena +  np.linalg.norm(X[j][i]- self.x_range[i][1])

            # 定義域でクリップ
            for i in range(len(X[0])):
                X[j][i] = np.clip(X[j][i],self.x_range[i][0],self.x_range[i][1])

            #オブジェクトとターゲットの位置の設定
            self.obj_pos = param[j,:2]
            self.goal_pos = param[j,2:]

            #軌道のオブジェクトからの距離
            dis_from_obj = 0.07
            goal1 = self.obj_pos

            # 軌道決定パラ
            if self.d == 3:
                goal1 = [(goal1[0] - np.sin(X[j,0])*dis_from_obj)*scale,(goal1[1] + np.cos(X[j,0])*dis_from_obj)*scale,0]
                goal2 = [X[j,1]*scale,X[j,2]*scale,0]
            elif self.d == 4:
                goal1 = [X[j,0]*scale,X[j,1]*scale,0]
                goal2 = [X[j,2]*scale,X[j,3]*scale,0]

            # 信号の取得
            dmp = DMP(goal1,goal2,scale)
            dmp.imitation()
            path = dmp.get_path()

            # 総時刻(行動回数)を指定
            T = len(path) + 30

            state = self.env.reset()
            # ターゲットとゴールの設定
            self.env.unwrapped.set_obj_pos(self.obj_pos)
            self.env.unwrapped.set_goal_pos(self.goal_pos)

            # 1エピソードのシミュレーション
            for t in range(T):
                if t < len(path):
                    action = np.append(path[t],0)
                else:
                    action = [0,0,0,0]
                
                # 状態を遷移
                next_state, reward, terminated, truncated, info = self.env.step(action)

            if not len(X) == 1:
                self.penas[j] = pena
            rewards[j] = -reward + pena
            print("{}:{}".format(self.eval_count,-reward + pena))

        self.env.close()
        return rewards
    
    def close_env(self):
        self.env.close()

    def get_pena(self):
        return self.penas.mean()
    
    def optimal_solution(self, param):
        # パラメータに対応する最適解（テスト用）
        # 計算ができなければNoneを返す
        return None
    
    def optimal_evaluation(self, param):
        # パラメータに対応する最良評価値（テスト用）
        # 計算ができなければNoneを返す
        return None
    
    def generate_context(self, lam):
        con_list = np.zeros((lam,self.param_dim))
        for i in range(len(con_list)):
            for j in range(len(con_list[0])):
                con_list[i][j] = np.random.uniform(self.param_range[j][0],self.param_range[j][1])
        return con_list


class NewBenchmark(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, 
                    d,                      # 次元数
                    max_eval=np.inf,        # 最大評価回数
                    param=None,             # パラメータ
                    param_range = [[-2,2]]  # パラメータの定義域
                ):
        super(NewBenchmark, self).__init__(max_eval, param, param_range)
        self.d = d          # 問題の次元数
        self.param_dim = 1  # パラメータの次元数

    def _evaluation(self, X, param):
        # (X, param)の評価値を返す
        return None
    
    def optimal_solution(self, param):
        # パラメータに対応する最適解（テスト用）
        # 計算ができなければNoneを返す
        return None
    
    def optimal_evaluation(self, param):
        # パラメータに対応する最良評価値（テスト用）
        # 計算ができなければNoneを返す
        return None

if __name__ == '__main__':
    bench = FechPush()
    X = np.array([[0,0,0],[0,0,0]])
    # param = np.array([[0.05,0,0,0],[0,0,0.05,0]])
    param = np.array([[0.05,0,0]])
    print(bench._evaluation(X,param))