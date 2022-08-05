'''
Ref: https://github.com/chanshing/xnes/blob/master/xnes.py
'''
from typing import Optional, List, Tuple, cast
from scipy.linalg import (det, expm)
from scipy.stats import multivariate_normal, norm
import numpy as np
from xbbo.initial_design import ALL_avaliable_design

from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trial, Trials
from . import alg_register


@alg_register.register('xnes')
class XNES(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 pop_size=None,
                 amat=1,
                 sample_method: str = 'Gaussian',
                 eta_sigma=None,
                 eta_mu=1.0,
                 eta_bmat=None,
                 use_adasam=False,
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   **kwargs)
        # Uniform2Gaussian.__init__(self, )

        # configs = self.space.get_hyperparameters()
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()
        if sample_method == 'Gaussian':
            pass
            # self.sampler = Gaussian_sampler(self.dimension, self.bounds,
            #                                 self.rng)
        else:
            raise NotImplementedError

        self.eyemat = np.eye(self.dimension)
        self.eta_mu = eta_mu
        self.s_try = []
        self.z_try = []
        self.f_try = []
        lb = self.bounds.lb
        ub = self.bounds.ub
        self.mu = 0.5 * (ub - lb) + lb
        amat = np.array(amat)
        if len(amat.shape) != 2:
            amat = np.eye(self.dimension)*amat / (np.mean(ub-lb)*10)
        sigma = np.abs(det(amat))**(1.0 / self.dimension)
        self.bmat = amat * (1.0 / sigma)
        self.sigma = sigma
        self.pop_size = int(
            4 + 3 * np.log(self.dimension)) if pop_size is None else pop_size
        self.eta_sigma = 3 * (3 + np.log(self.dimension)) * (
            1.0 / (5 * self.dimension * np.sqrt(self.dimension))
        ) if eta_sigma is None else eta_sigma
        self.eta_bmat = 3 * (3 + np.log(self.dimension)) * (
            1.0 / (5 * self.dimension *
                   np.sqrt(self.dimension))) if eta_bmat is None else eta_bmat
        self.trials = Trials(space, dim=self.dimension)

        self.use_adasam = use_adasam
        self.eta_sigma_init = self.eta_sigma

        self.utilities = self._fit_shaping()

    def _fit_shaping(self, ):
        a = np.log(1 + 0.5 * self.pop_size)
        utilities = np.clip(
            [a - np.log(k) for k in range(1, self.pop_size + 1)], 0, None)
        utilities /= sum(utilities)
        utilities -= 1.0 / self.pop_size  # broadcast
        return utilities

    def _suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            s_try = self.rng.randn(self.dimension)
            self.s_try.append(s_try)
            z_try = self.mu + self.sigma * np.dot(s_try,
                                                  self.bmat)  # broadcast
            z_try_ = np.clip(z_try, self.bounds.lb,
                                     self.bounds.ub)
            self.z_try.append(z_try)
            config = DenseConfiguration.from_array(self.space, z_try_)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=z_try,
                      origin='XNES'))

        return trial_list

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            self.f_try.append(trial.observe_value)
        if len(self.f_try) < self.pop_size:
            return

        s_try = np.asarray(self.s_try)
        z_try = np.asarray(self.z_try)
        f_try = np.asarray(self.f_try)

        isort = np.argsort(f_try)
        f_try = f_try[isort]
        s_try = s_try[isort]
        z_try = z_try[isort]

        u_try = self.utilities

        if self.use_adasam and sigma_old is not None:  # sigma_old must be available
            self.eta_sigma = self.adasam(self.eta_sigma, self.mu, self.sigma, self.bmat,
                                    sigma_old, z_try)

        dj_delta = np.dot(u_try, s_try)
        dj_mmat = np.dot(
            s_try.T,
            s_try * u_try.reshape(self.pop_size, 1)) - sum(u_try) * self.eyemat
        dj_sigma = np.trace(dj_mmat) * (1.0 / self.dimension)
        dj_bmat = dj_mmat - dj_sigma * self.eyemat

        sigma_old = self.sigma

        # update
        self.mu += self.eta_mu * self.sigma * np.dot(self.bmat, dj_delta)
        self.sigma *= np.exp(0.5 * self.eta_sigma * dj_sigma)
        self.bmat = np.dot(self.bmat, expm(0.5 * self.eta_bmat * dj_bmat))

        # # logging
        # self.history['fitness'].append(fitness)
        # self.history['sigma'].append(sigma)
        # self.history['eta_sigma'].append(eta_sigma)

        self.s_try = []
        self.z_try = []
        self.f_try = []

    def adasam(self, eta_sigma, mu, sigma, bmat, sigma_old, z_try):
        """ Adaptation sampling """
        eta_sigma_init = self.eta_sigma_init
        dim = self.dimension
        c = .1
        rho = 0.5 - 1./(3*(dim+1))  # empirical

        bbmat = np.dot(bmat.T, bmat)
        cov = sigma**2 * bbmat
        sigma_ = sigma * np.sqrt(sigma*(1./sigma_old))  # increase by 1.5
        cov_ = sigma_**2 * bbmat

        p0 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov)
        p1 = multivariate_normal.logpdf(z_try, mean=mu, cov=cov_)
        w = np.exp(p1-p0)

        # Mann-Whitney. It is assumed z_try was in ascending order.
        n = self.pop_size
        n_ = sum(w)
        u_ = sum(w * (np.arange(n)+0.5))

        u_mu = n*n_*0.5
        u_sigma = np.sqrt(n*n_*(n+n_+1)/12.)
        cum = norm.cdf(u_, loc=u_mu, scale=u_sigma)

        if cum < rho:
            return (1-c)*eta_sigma + c*eta_sigma_init
        else:
            return min(1, (1+c)*eta_sigma)

opt_class = XNES
