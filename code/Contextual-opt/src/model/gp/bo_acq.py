#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
# from GPyOpt.util.general import get_quantiles

class AcquisitionVariance(AcquisitionBase):

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionVariance, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter

    @staticmethod
    def fromConfig(model, space, optimizer, cost_withGradients, config):
        return AcquisitionVariance(model, space, optimizer, cost_withGradients, jitter=config['jitter'])

    def _compute_acq(self, x):
        """
        Computes the Variance per unit of cost
        """
        _, s = self.model.predict(x)
        return np.sum(s)
        # return np.prod(s)

    def _compute_acq_withGradients(self, x):
        _, s, _, dsdx = self.model.predict_withGradients(x)
        if dsdx.ndim == 2:
            dsdx = dsdx.sum(axis=0)
        return np.sum(s), dsdx
        # return np.prod(s, axis=0), np.sum(np.prod(s) * dsdx / s, axis=0)