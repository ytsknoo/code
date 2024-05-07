#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base import ParametrizedObjectiveFunction
import numpy as np
import copy
import math

# public symbols
__all__ = ['ShiftSphere', 'ShiftRosenbrock', 'ShiftEllipsoid']
ips = 0.25

# 乱数生成用関数
def randn_from_rand(size=[1,1]):
    np.random.seed(101)

    randn_len = 1
    randn_list = np.empty([0])
    randn_matrix = np.zeros((size[0],size[1]))

    for i in range(len(size)):
        randn_len *= size[i]
    
    for j in range(int(randn_len/2) + 1):
        rand = np.random.rand(2)

        #Use box-muller to get normally distributed random numbers
        randn = np.zeros(2)
        randn[0] = np.sqrt(-2.*np.log(rand[0]))*np.cos(2*np.pi*rand[1])
        randn[1] = np.sqrt(-2.*np.log(rand[0]))*np.sin(2*np.pi*rand[1])

        randn_list = np.append(randn_list,randn)
    
    m = 0
    for k in range(size[0]):
        for l in range(size[1]):
            randn_matrix[k,l] = randn_list[m]
            m += 1


    return randn_matrix


class ShiftSphere(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(ShiftSphere, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        # self.G = randn_from_rand(size=[self.param_dim,d]).T
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        Y = X - param_expan
        evals = (Y**2).sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        return np.dot(self.G,param).reshape(1,self.d)
    
    def optimal_evaluation(self, param):
        return 0.
    
    def generate_context(self, lam):
        return np.random.rand(lam, self.param_dim) * 4 - 2

class NonLinearShiftSphere(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(NonLinearShiftSphere, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        # self.G = randn_from_rand(size=[self.param_dim,d]).T
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        Y = X - param_expan
        evals = (Y**2).sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        return np.dot(self.G,param**2).reshape(1,self.d)
    
    def optimal_evaluation(self, param):
        return 0.

class NoisedShiftSphere(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(NoisedShiftSphere, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        # self.G = randn_from_rand(size=[self.param_dim,d]).T
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)

        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        Y = X - param_expan + np.random.randn(1,self.d)*ips
        np.random.set_state(randstate)
        evals = (Y**2).sum(axis=1)

        return evals
    
    def optimal_solution(self, param):
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        solution = np.dot(self.G,param).reshape(1,self.d) + np.random.randn(1,self.d)*ips
        print(np.dot(self.G,param).reshape(1,self.d))
        np.random.set_state(randstate)
        return solution
    
    def optimal_evaluation(self, param):
        return 0.

class RotateSphere(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[0,math.pi*2]]):
        super(RotateSphere, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        if self.d == 2:
            G = np.array([[math.cos(param),-math.sin(param)],
                        [math.sin(param),math.cos(param)]]) #回転行列
        else:
            G = np.eye(self.d)
            G[0:2,0:2] = np.array([[math.cos(param),-math.sin(param)],
                        [math.sin(param),math.cos(param)]]) #回転行列

        E = np.zeros(self.d)
        E[0] = 1
        
        Y = np.zeros_like(X)

        for i in range(len(X)):
            # X[i] = X[i] - E
            Y[i] = np.dot(X[i],G.T) - E

        evals = (Y**2).sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        if self.d == 2:
            G = np.array([[math.cos(param),-math.sin(param)],
                        [math.sin(param),math.cos(param)]]) #回転行列
        else:
            G = np.eye(self.d)
            G[0:2,0:2] = np.array([[math.cos(param),-math.sin(param)],
                        [math.sin(param),math.cos(param)]]) #回転行列
        
        E = np.zeros(self.d)
        E[0] = -1
        return np.dot(E,G.T)
    
    def optimal_evaluation(self, param):
        return 0.


class ShiftRastrigin(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(ShiftRastrigin, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        Y = X - param_expan
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if -5.12 <= Y[i][j] <= 5.12:
                    Y[i][j] = 10 + (Y[i][j]**2 - 10 * math.cos(2*math.pi*Y[i][j]))
                else:
                    Y[i][j] = 10 + (5.12**2 - 10 * math.cos(2*math.pi*5.12))
        evals = Y.sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        return param_expan
    
    def optimal_evaluation(self, param):
        return 0.

class NonLinearShiftRastrigin(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(NonLinearShiftRastrigin, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        Y = X - param_expan
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if -5.12 <= Y[i][j] <= 5.12:
                    Y[i][j] = 10 + (Y[i][j]**2 - 10 * math.cos(2*math.pi*Y[i][j]))
                else:
                    Y[i][j] = 10 + (5.12**2 - 10 * math.cos(2*math.pi*5.12))
        evals = Y.sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        return param_expan
    
    def optimal_evaluation(self, param):
        return 0.

class NoisedShiftRastrigin(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(NoisedShiftRastrigin, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        # self.G = randn_from_rand(size=[self.param_dim,d]).T
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        Y = X - param_expan + np.random.randn(1,self.d)*ips
        np.random.set_state(randstate)
        for i in range(len(Y)):
            for j in range(len(Y[i])):
                if -5.12 <= Y[i][j] <= 5.12:
                    Y[i][j] = 10 + (Y[i][j]**2 - 10 * math.cos(2*math.pi*Y[i][j]))
                else:
                    Y[i][j] = 10 + (5.12**2 - 10 * math.cos(2*math.pi*5.12))
        evals = Y.sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        solution = np.dot(self.G,param).reshape(1,self.d) + np.random.randn(1,self.d)*ips
        np.random.set_state(randstate)
        return solution
    
    def optimal_evaluation(self, param):
        return 0.


class ShiftEasom(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(ShiftEasom, self).__init__(max_eval, param, param_range)
        self.d = 2
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        Y = X - param_expan
        evals = np.zeros(len(Y),)
        for i in range(len(Y)):
            evals[i] = -np.cos(Y[i][0])*np.cos(Y[i][1])*np.exp(-Y[i][1]**2 + 2*np.pi*Y[i][1] - Y[i][0]**2 + 2*np.pi*Y[i][0] - 2*np.pi**2) + 1
        return evals
    
    def optimal_solution(self, param):
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        return param_expan + np.pi
    
    def optimal_evaluation(self, param):
        return 0.

class NonLinearShiftEasom(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(NonLinearShiftEasom, self).__init__(max_eval, param, param_range)
        self.d = 2
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        Y = X - param_expan
        evals = np.zeros(len(Y),)
        for i in range(len(Y)):
            evals[i] = -np.cos(Y[i][0])*np.cos(Y[i][1])*np.exp(-Y[i][1]**2 + 2*np.pi*Y[i][1] - Y[i][0]**2 + 2*np.pi*Y[i][0] - 2*np.pi**2) + 1
        return evals
    
    def optimal_solution(self, param):
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        return param_expan + np.pi
    
    def optimal_evaluation(self, param):
        return 0.

class NoisedShiftEasom(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        super(NoisedShiftEasom, self).__init__(max_eval, param, param_range)
        self.d = 2
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        # Y = X - param * np.ones_like(X)
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        Y = X - param_expan + np.random.randn(1,self.d)*ips
        np.random.set_state(randstate)
        evals = np.zeros(len(Y),)
        for i in range(len(Y)):
            evals[i] = -np.cos(Y[i][0])*np.cos(Y[i][1])*np.exp(-Y[i][1]**2 + 2*np.pi*Y[i][1] - Y[i][0]**2 + 2*np.pi*Y[i][0] - 2*np.pi**2) + 1
        return evals
    
    def optimal_solution(self, param):
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        solution = np.dot(self.G,param).reshape(1,self.d) + np.random.randn(1,self.d)*ips + np.pi
        np.random.set_state(randstate)
        return solution
    
    def optimal_evaluation(self, param):
        return 0.



class ShiftRosenbrock(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        """
        Rosenbrock function
        """
        super(ShiftRosenbrock, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        Y = X - param_expan
        # Y = X - param * np.ones_like(X)
        evals = np.sum(100 * (Y[:, :-1]**2 - Y[:, 1:])**2 + (Y[:, :-1] - 1.)**2, axis=1)
        return evals

    def optimal_solution(self, param):
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        # return (param + 1)* np.ones(self.d)
        return param_expan + 1
    
    def optimal_evaluation(self, param):
        return 0.

class NonLinearRosenbrock(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        """
        Rosenbrock function
        """
        super(NonLinearRosenbrock, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        Y = X - param_expan
        # Y = X - param * np.ones_like(X)
        evals = np.sum(100 * (Y[:, :-1]**2 - Y[:, 1:])**2 + (Y[:, :-1] - 1.)**2, axis=1)
        return evals

    def optimal_solution(self, param):
        param_expan = np.dot(self.G,param**2).reshape(1,self.d)
        # return (param + 1)* np.ones(self.d)
        return param_expan + 1
    
    def optimal_evaluation(self, param):
        return 0.

class NoisedRosenbrock(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2],[-2,2]]):
        """
        Rosenbrock function
        """
        super(NoisedRosenbrock, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 2
        # self.G = randn_from_rand(size=[self.param_dim,d]).T
        self.G = np.random.randn(d,self.param_dim)

    def _evaluation(self, X, param):
        param = param.reshape(self.param_dim,1) #文脈ベクトル縦に
        param_expan = np.dot(self.G,param).reshape(1,self.d)
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        Y = X - param_expan + np.random.randn(1,self.d)*ips
        np.random.set_state(randstate)
        # Y = X - param * np.ones_like(X)
        evals = np.sum(100 * (Y[:, :-1]**2 - Y[:, 1:])**2 + (Y[:, :-1] - 1.)**2, axis=1)
        return evals

    def optimal_solution(self, param):
        randstate = np.random.get_state()
        np.random.seed(int(abs(sum(param))*10e4))
        solution = np.dot(self.G,param).reshape(1,self.d) + np.random.randn(1,self.d)*ips + 1
        np.random.set_state(randstate)
        return solution
    
    def optimal_evaluation(self, param):
        return 0.


class ShiftEllipsoid(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2]]):
        super(ShiftEllipsoid, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1
        self.coefficient = 1000 ** (np.arange(d) / float(d - 1))

    def _evaluation(self, X, param):
        Y = X - param * np.ones_like(X)
        tmp = Y * self.coefficient
        evals = (tmp**2).sum(axis=1)
        return evals

    def optimal_solution(self, param):
        return param * np.ones(self.d)
    
    def optimal_evaluation(self, param):
        return 0.


class AffineSphere(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[1,5]]):
        super(AffineSphere, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1
        assert self.d > 1 

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """

        coefficient_param = param ** (np.arange(self.d) / float(self.d - 1))
        Y = X - np.ones_like(X) + np.ones_like(X) * coefficient_param
        evals = (Y**2).sum(axis=1)
        return evals

    def optimal_solution(self, param):
        coefficient_param = param ** (np.arange(self.d) / float(self.d - 1))
        return np.ones(self.d) - np.ones(self.d) * coefficient_param
    
    def optimal_evaluation(self, param):
        return 0.

class AffineRosenbrock(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[1,5]]):
        """
        Rosenbrock function
        """
        super(AffineRosenbrock, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1
        assert self.d > 1 

    def _evaluation(self, X, param):
        coefficient_param = param ** (np.arange(self.d) / float(self.d - 1))
        Y = X + np.ones_like(X) - np.ones_like(X) * coefficient_param
        evals = np.sum(100 * (Y[:, :-1]**2 - Y[:, 1:])**2 + (Y[:, :-1] - 1.)**2, axis=1)
        return evals

    def optimal_solution(self, param):
        coefficient_param = param ** (np.arange(self.d) / float(self.d - 1))
        return np.ones(self.d) * coefficient_param
    
    def optimal_evaluation(self, param):
        return 0.

class AffineEllipsoid(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[1,5]]):
        super(AffineEllipsoid, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1
        self.coefficient = 1000 ** (np.arange(d) / float(d - 1))
        assert self.d > 1 

    def _evaluation(self, X, param):
        coefficient_param = param ** (np.arange(self.d) / float(self.d - 1))
        Y = X - np.ones_like(X) + np.ones_like(X) * coefficient_param
        tmp = Y * self.coefficient
        evals = (tmp**2).sum(axis=1)
        return evals

    def optimal_solution(self, param):
        coefficient_param = param ** (np.arange(self.d) / float(self.d - 1))
        return np.ones(self.d) - np.ones(self.d) * coefficient_param
    
    def optimal_evaluation(self, param):
        return 0.


class CosSphere(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2]]):
        super(CosSphere, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1

    def _evaluation(self, X, param):
        """
        :param X: candidate solutions
        :type X: array_like, shape=(lam, d), dtype=float
        :return: evaluation values
        :rtype: array_like, shape=(lam), dtype=float
        """
        Y = X - np.cos(param * np.arange(1,self.d+1))
        evals = (Y**2).sum(axis=1)
        return evals
    
    def optimal_solution(self, param):
        return np.cos(param * np.arange(1,self.d+1))
    
    def optimal_evaluation(self, param):
        return 0.


class CosRosenbrock(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[-2,2]]):
        """
        Rosenbrock function
        """
        super(CosRosenbrock, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1

    def _evaluation(self, X, param):
        Y = X - np.cos(param * np.arange(1,self.d+1))
        evals = np.sum(100 * (Y[:, :-1]**2 - Y[:, 1:])**2 + (Y[:, :-1] - 1.)**2, axis=1)
        return evals

    def optimal_solution(self, param):
        return np.cos(param * np.arange(1,self.d+1)) + np.ones(self.d)
    
    def optimal_evaluation(self, param):
        return 0.

class CosEllipsoid(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[1,5]]):
        super(CosEllipsoid, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1
        self.coefficient = 1000 ** (np.arange(d) / float(d - 1))

    def _evaluation(self, X, param):
        Y = X - np.log(param * np.arange(1,self.d+1))
        tmp = Y * self.coefficient
        evals = (tmp**2).sum(axis=1)
        return evals

    def optimal_solution(self, param):
        return np.log(param * np.arange(1,self.d+1))
    
    def optimal_evaluation(self, param):
        return 0.



class SMDLower(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, 
                    smd,                    # SMD class
                    max_eval=np.inf,        # 最大評価回数
                    param=None,             # パラメータ
                ):
        super(SMDLower, self).__init__(max_eval, param, param_range=None)
        self.smd = smd
        self.d = self.smd.d_lower         # 問題の次元数
        self.param_dim = self.smd.d_upper  # パラメータの次元数
        self.param_range = np.c_[self.smd.upper_domain()]

    def _evaluation(self, X, param):
        # (X, param)の評価値を返す
        if param.ndim == 1:
            param = param[None,:]
        evals = self.smd._evaluation_lower(param, X)
        return evals

    def optimal_solution(self, param):
        return self.smd.lower_optimal_solution(param[None,:])[0]
    
    def optimal_evaluation(self, param):
        # パラメータに対応する最良評価値（テスト用）
        # 計算ができなければNoneを返す
        return self.smd._evaluation_lower(param[None,:], self.optimal_solution(param)[None,:])[0]
    
    def _update_best_eval(self, X, evals):

        # 定義域内でない解は最良解にしない
        constraint_lower, constraint_upper = self.smd.lower_domain()
        is_out = (np.max(np.stack((
                X - constraint_upper, 
                constraint_lower - X, 
                np.zeros_like(X))), axis=0) ** 2).sum(axis=1) > 0
        penalty = np.zeros(len(X)) 
        penalty[is_out == True] = np.inf
        evals_penalty = evals + penalty

        if self.best_solution is None or self.is_better(self.get_best(evals_penalty), self.best_eval):
            self.best_eval = self.get_best(evals_penalty)
            self.best_solution = X[self.get_best_idx(evals_penalty)]

class RotateEllipsoid(ParametrizedObjectiveFunction):
    minimization_problem = True

    def __init__(self, d, max_eval=np.inf, param=None, param_range = [[0,math.pi]]):
        super(RotateEllipsoid, self).__init__(max_eval, param, param_range)
        self.d = d
        self.param_dim = 1
        self.coefficient = 1000 ** (np.arange(d) / float(d - 1))

    def _evaluation(self, X, param):

        G = np.array([[math.cos(param),-math.sin(param)],
                      [math.sin(param),math.cos(param)]]) #回転行列
        
        Y = np.zeros_like(X)

        for i in range(len(X)):
            Y[i] = np.dot(G,X[i])

        tmp = Y * self.coefficient
        evals = (tmp**2).sum(axis=1)
        return evals

    def optimal_solution(self, param):
        return np.zeros_like(self.d)
    
    def optimal_evaluation(self, param):
        return 0.


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
    bench = RotateSphere(2)
    X = np.array([[0,0],[0,0]])
    param = np.array([[math.pi],[math.pi]])
    print(bench._evaluation(X,param))