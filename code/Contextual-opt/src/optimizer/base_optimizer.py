#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import abstractmethod
import numpy as np

# public symbols
__all__ = ['BaseOptimizer']


class BaseOptimizer(object):
    """
    Base class for information geometric util
    """

    @abstractmethod
    def sampling_model(self):
        pass

    @abstractmethod
    def update(self, X, evals):
        """
        Abstract method for parameter updating.
        """
        pass

    def terminate_condition(self):
        """
        Check terminate condition.
        :return bool: terminate condition is satisfied or not
        """
        return False

    def verbose_display(self):
        """
        Return verbose display string.
        :return str: string for verbose display
        """
        return ''

    def log_header(self):
        """
        Return log header list.
        :return: header info list for log
        :rtype string list:
        """
        return []

    def log(self):
        """
        Return log string list.
        :return: log string list
        :rtype string list:
        """
        return []

    def run(self, sampler, logger=None, verbose=True):
        """
        Running script of Information Geometric Optimization (IGO)
        """

        f = sampler.f
        self.objective_func = sampler.f
        
        if logger is not None:
            if hasattr(sampler.f,"get_pena"):
                logger.write_csv(['Ite'] + f.info_header() + self.log_header() + sampler.log_header() + ['penalty'])
            else :
                logger.write_csv(['Ite'] + f.info_header() + self.log_header() + sampler.log_header())


        ite = 0

        # テストで追加
        self.eval_cons_flag = 0
        dif_evals = 1e-4

        # self.x_log = np.array([])
        # self.eval_log = np.array([])

        while not sampler.f.terminate_condition() and not self.terminate_condition():
            ite += 1

            # sampling and evaluation
            X, evals = sampler(self.sampling_model())


            # if ite == 1:
            #     self.x_log = X
            #     self.eval_log = evals
            # else:
            #     self.x_log = np.append(self.x_log,X,axis=0)
            #     self.eval_log = np.append(self.eval_log,evals)

            if evals.max() - evals.min() < dif_evals:
                self.eval_cons_flag = self.eval_cons_flag + len(evals)

            # display and save log
            if verbose:
                print(str(ite) + f.verbose_display() + self.verbose_display() + sampler.verbose_display())
            if logger is not None:
                if hasattr(sampler.f,"get_pena"):
                    logger.write_csv([str(ite)] + f.info_list() + self.log() + sampler.log() + [str(sampler.f.get_pena())])
                else :
                    logger.write_csv([str(ite)] + f.info_list() + self.log() + sampler.log())

            # parameter update
            self.update(X, evals)

        return [f.eval_count, f.best_eval]
    