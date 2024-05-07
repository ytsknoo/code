#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xmlrpc.client import Boolean
import numpy as np
from GPy.kern import Kern, RBF, Bias, Linear, Coregionalize, Matern52, Matern32
from GPy.models import GPRegression, GPMultioutRegression, GPCoregionalizedRegression
from GPy.util.multioutput import LCM, ICM
from GPy.core.parameterization.priors import Prior
from sklearn.neighbors import KernelDensity
from scipy.stats import rankdata
import warnings
import itertools 
from GPy.inference.mcmc import HMC

from GPyOpt.models.gpmodel import GPModel

class GPInterface(GPModel):

    kernel: Kern
    model: GPRegression
    kernel_lengthscale_prior: Prior
    kernel_variance_prior: Prior
    likelihood_variance_prior: Prior

    invariant_transform = False

    def __init__(
        self,
        d: int,
        kernel=None,
        **kwargs
    ):
        super(GPInterface, self).__init__(kernel=kernel, **kwargs)

        self.d = d
        self.model = None
        self.kernel_lengthscale_prior = None
        self.kernel_variance_prior = None
        self.likelihood_variance_prior = None

        self.kernel = kernel
        
        self.Y_mean = 0.0
        self.Y_std = 1.0
        self.output_num = None

    def _create_model(self, X: np.ndarray, Y: np.ndarray, normalize=False):
        """
        create gaussian process regression model.

        Args:
            X: observed search points
            Y: observed evaluation values
        """

        if Y.shape[1] == 1:
            self._create_single_output_model(X, Y, normalize)
        else:
            self._create_multiple_output_model(X, Y, normalize)
        
        self.X_data = X
        self.Y_data = Y
        self.normalize = normalize
        

    def _create_single_output_model(self, X: np.ndarray, Y: np.ndarray, normalize: Boolean = True):
        """
        create gaussian process regression model with single output.

        Args:
            X: observed search points
            Y: observed evaluation values
        """
        self.output_num = 1
        if normalize:
            self.Y_mean = np.mean(Y)
            self.Y_std = np.std(Y)   # not normalize when using acquisition function based on std
            Y = (Y - self.Y_mean) / self.Y_std
        
        if self.kernel is None:
            self.kernel = RBF(self.d)

        self.model = GPRegression(
            X, Y, kernel=self.kernel, 
        )

        if self.kernel_lengthscale_prior is not None:
            self.model.kern.lengthscale.set_prior(self.kernel_lengthscale_prior)
        if self.kernel_variance_prior is not None:
            self.model.kern.variance.set_prior(self.kernel_variance_prior)
        if self.likelihood_variance_prior is not None:
            self.model.likelihood.variance.set_prior(self.likelihood_variance_prior)

        # self.model.optimize()

        self.model.optimize_restarts(num_restarts=5, max_iters=1000, verbose=self.verbose)

        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)
    
    def _create_multiple_output_model(self, X: np.ndarray, Y: list, normalize: Boolean = True):
        """
        create gaussian process regression model with multiple output.

        Args:
            X: observed search points
            Y: observed evaluation values
        """
        self.output_num = Y.shape[0]

        if normalize:
            self.Y_mean = np.mean(Y, axis=1)[:,None,:]
            self.Y_std = np.std(Y, axis=1)[:,None,:]   # not normalize when using acquisition function based on std
            Y = (Y - self.Y_mean) / self.Y_std
        
        # _X = np.vstack([np.c_[X, np.ones(X.shape[0]) * i] for i in range(self.output_num)])
        # _X = np.r_[[X.T for _ in range(self.output_num)]] #np.vstack([np.c_[X[i], [i]] for i in range(self.output_num)])
        _X = X
        _Y = Y #.flatten()[:, np.newaxis]

        if self.kernel is None:
            kernels_list = [RBF(self.d), Linear(self.d), Matern52(self.d)]
            self.kernel = LCM(input_dim=self.d, num_outputs=self.output_num, kernels_list=kernels_list, W_rank=1)
            # self.kernel = ICM(input_dim=self.d, num_outputs=self.output_num, kernel=self.kernel, W_rank=1)

        # self.model = GPRegression( _X[0], _Y[0], kernel=self.kernel, noise_var=0.1 )
        self.model = GPCoregionalizedRegression(_X, _Y, kernel=self.kernel)
        
        # if self.kernel_lengthscale_prior is not None:
        #     self.model.kern.lengthscale.set_prior(self.kernel_lengthscale_prior)
        # if self.kernel_variance_prior is not None:
        #     self.model.kern.variance.set_prior(self.kernel_variance_prior)
        # if self.likelihood_variance_prior is not None:
        #     self.model.likelihood.variance.set_prior(self.likelihood_variance_prior)

        # self.model.optimize()
        self.model.optimize_restarts(num_restarts=5, max_iters=1000, verbose=False)

    def add_data(self, X_new, Y_new):

        if self.output_num == 1:
            # if X_new.ndim == 1:
            #     X_new = X_new[:,None]
            # if Y_new.ndim == 1:
            #     Y_new = Y_new[:,None]
                
            self.X_data = np.vstack((self.X_data, X_new))
            self.Y_data = np.vstack((self.Y_data, Y_new))
        else:
            if X_new.ndim == 2:
                X_new = X_new[:,:,None]
            if Y_new.ndim == 2:
                Y_new = Y_new[:,:,None,]

            self.X_data = np.hstack([self.X_data, X_new])
            self.Y_data = np.hstack([self.Y_data, Y_new])

        self._create_model(self.X_data, self.Y_data, self.normalize)

    def set_kernel_param_prior(
        self,
        prior: Prior,
        kernel_lengthscale=True,
        kernel_variance=True,
        likelihood_variance=True,
    ):
        """
        set prior distribution of hyperparameter. This should be called before creating GP model.
        """
        if kernel_lengthscale:
            self.set_kernel_lengthscale_prior(prior)
        if kernel_variance:
            self.set_kernel_variance_prior(prior)
        if likelihood_variance:
            self.set_likelihood_variance_prior(prior)

    def set_kernel_lengthscale_prior(self, prior):
        self.kernel_lengthscale_prior = prior

    def set_kernel_variance_prior(self, prior):
        self.kernel_variance_prior = prior

    def set_likelihood_variance_prior(self, prior):
        self.likelihood_variance_prior = prior

    def predict_mean(self, X, include_likelihood=False):
        """
        predict the evaluation value at given points using created GP model
        """
        
        if self.output_num != 1:

            data_num = X.shape[0]
            X = np.vstack([np.c_[X, np.ones(X.shape[0]) * i] for i in range(self.output_num)])

            Y_metadata = {'output_index':X[:,-1:].astype(int)}
            model_mean = self.model.predict(
                    X, include_likelihood=include_likelihood, Y_metadata=Y_metadata
                )[0] 

            # index = np.hstack([np.ones(data_num) * i for i in range(self.output_num)]).astype(int)

            if self.normalize:
                _mean = np.vstack([self.Y_mean[:,0] for i in range(data_num)])
                _std = np.vstack([self.Y_std[:,0] for i in range(data_num)])
                model_mean = model_mean * _std + _mean

            return model_mean[:,0].reshape((self.output_num, data_num)) 

            # return model_mean.reshape((self.output_num, data_num)).T
            
        else:
            return self.model.predict(
                    X, include_likelihood=include_likelihood
                )[0] * self.Y_std + self.Y_mean

    def predict_sigma(self, X, include_likelihood=False):
        """
        predict the evaluation value at given points using created GP model
        """
        if self.output_num != 1:
            data_num = X.shape[0]

            X = np.vstack([np.c_[X, np.ones(X.shape[0]) * i] for i in range(self.output_num)])

            Y_metadata = {'output_index':X[:,-1:].astype(int)}
            model_diag_std = np.sqrt(self.model.predict(
                            X, full_cov=False, include_likelihood=include_likelihood, Y_metadata=Y_metadata
                        )[1]) 
            
            if self.normalize:
                _std = np.vstack([self.Y_std[:,0] for i in range(data_num)])
                model_diag_std  = model_diag_std * _std
            
            # index = np.hstack([np.ones(data_num) * i for i in range(self.output_num)]).astype(int)

            return model_diag_std[:,0].reshape((self.output_num, data_num)) 
            # return (model_diag_std * self.Y_std).reshape((self.output_num,-1))
        
        else:
            model_diag_std = np.sqrt(self.model.predict(
                            X, full_cov=False, include_likelihood=include_likelihood
                        )[1]) 
            
            return model_diag_std * self.Y_std
    
    def predict(self, X, with_noise=False):
        return self.predict_mean(X, include_likelihood=with_noise), self.predict_sigma(X, include_likelihood=with_noise)

    def predict_withGradients(self, X):
        
        # if X.ndim==1: 
        #     X = X[None,:]
        m, v = self.predict(X) 

        if self.output_num != 1:
            X = np.vstack([np.c_[X, np.ones(X.shape[0]) * i] for i in range(self.output_num)])

        dmdx, dvdx = self.model.predictive_gradients(X)

        if self.output_num != 1:
            dmdx = dmdx[:,:,0]
            dsdx = dvdx / (2*np.sqrt(v)) 

            # if self.normalize:
            #     dmdx *= self.Y_std[:,0]
                # dsdx *= self.Y_std[:,0]
        else:
            dmdx = dmdx[:,:,0]
            dsdx = dvdx / (2*np.sqrt(v)) 

            # if self.normalize:
            #     dmdx *= self.Y_std
                # dsdx /= self.Y_std

        if self.output_num == 1:
            return m, np.sqrt(v), dmdx, dsdx
        else:
            return m, np.sqrt(v), dmdx[:,:-1], dsdx[:,:-1]
    
    def predict_covariance(self, X, include_likelihood=False):
        if self.output_num != 1:
            data_num = X.shape[0]
            input_dim = X.shape[1]
            X = np.vstack([np.c_[X, np.ones(X.shape[0]) * i] for i in range(self.output_num)])

            # Y_metadata = {'output_index':X[:,-1:].astype(int)}
            # print(X.shape)
            # exit()
            # model_cov = self.model.posterior_covariance_between_points(
            #                 X, X, include_likelihood=include_likelihood, Y_metadata=Y_metadata
            #             )[1]
            
            # print(model_cov.shape)

            # cov_list = np.zeros((data_num, self.output_num,self.output_num))
            
            # for i in range(data_num):
            #     idxs = [i * data_num + j for j in range(self.output_num)]
            #     cov_list[i] = model_cov[idxs,idxs] #* np.outer(self.Y_std, self.Y_std)

            # return cov_list

            cov_list = np.zeros((data_num, self.output_num,self.output_num))

            for data_id in range(data_num):
                _X = X[data_id::data_num]
                
                Y_metadata = {'output_index':_X[:,-1:].astype(int)}

                model_cov = self.model.posterior_covariance_between_points(
                            _X, _X, include_likelihood=include_likelihood, Y_metadata=Y_metadata
                        )

                if self.normalize:
                    model_cov = model_cov * np.outer(self.Y_std[:,0], self.Y_std[:,0])

                cov_list[data_id] = model_cov 
                
            return cov_list 

        else:
            model_cov = self.model.predict(
                            X, full_cov=True, include_likelihood=include_likelihood
                        )[1]
            
            return model_cov * (self.Y_std ** 2)