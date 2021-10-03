import gpytorch
import torch
from scipy.linalg import solve_triangular, cholesky
from scipy import stats
import numpy as np
import GPy
from sklearn import gaussian_process
# from botorch.acquisition import ExpectedImprovement
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch import fit_gpytorch_model
from botorch.optim import optimize_acqf
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan

from bbomark.surrogate.base import Surrogate


class GaussianTransform:
    """
    Transform data into Gaussian by applying psi = Phi^{-1} o F where F is the truncated ECDF.
    :param y: shape (n, dim)
    """

    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.sorted = y.copy()
        self.sorted.sort(axis=0)

    @staticmethod
    def z_transform(series, values_sorted=None):
        # applies truncated ECDF then inverse Gaussian CDF.
        if values_sorted is None:
            values_sorted = sorted(series)

        def winsorized_delta(n):
            return 1.0 / (4.0 * n ** 0.25 * np.sqrt(np.pi * np.log(n)))

        delta = winsorized_delta(len(series))

        def quantile(values_sorted, values_to_insert, delta):
            res = np.searchsorted(values_sorted, values_to_insert) / len(values_sorted)
            return np.clip(res, a_min=delta, a_max=1 - delta)

        quantiles = quantile(
            values_sorted,
            series,
            delta
        )

        quantiles = np.clip(quantiles, a_min=delta, a_max=1 - delta)

        return stats.norm.ppf(quantiles)

    def transform(self, y: np.array):
        """
        :param y: shape (n, dim)
        :return: shape (n, dim), distributed along a normal
        """
        assert y.shape[1] == self.dim
        # compute truncated quantile, apply gaussian inv cdf
        return np.stack([
            self.z_transform(y[:, i], self.sorted[:, i])
            for i in range(self.dim)
        ]).T


class StandardTransform:

    def __init__(self, y: np.array):
        assert y.ndim == 2
        self.dim = y.shape[1]
        self.mean = y.mean(axis=0, keepdims=True)
        self.std = y.std(axis=0, keepdims=True)

    def transform(self, y: np.array):
        z = (y - self.mean) / np.clip(self.std, a_min=0.001, a_max=None)
        return z


class SEkernel():
    def __init__(self):
        self.initialize()

    def initialize(self):
        # self.sumF = 0.001
        # self.sumL = 0.001
        # self.sumY = 0.001
        self.sigma_f = 1
        self.sigma_l = 1 # TODO 之前设的是1
        self.sigma_y = 0.001



    def compute_kernel(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            x2 = np.atleast_2d(x2)
            x1 = np.atleast_2d(x1)
            # noise = np.diag([self.sigma_y**2 for _ in range(x1.shape[0])])
            noise = np.eye(x1.shape[0]) * self.sigma_y**2
        else:
            x2 = np.atleast_2d(x2)
            x1 = np.atleast_2d(x1)
            noise = 0
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * (x1 @ x2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.sigma_l ** 2 * dist_matrix) + noise


class GaussianProcessRegressorARD_gpy(Surrogate):
    def __init__(self, dim, min_sample=3):
        super(GaussianProcessRegressorARD_gpy, self).__init__(dim, min_sample)
        self.cached = {}
        self.cached_mu_sigma = {}
        self.cached_mu_cov = {}
        self.kernel = GPy.kern.Matern52(input_dim=dim, ARD=True)
        # self.kernel = GPy.kern.RBF(input_dim=self.dim,
        #                       variance=0.001,
        #                       lengthscale=0.5,
        #                       ARD=True)

        self.is_fited = False
        self.standardlize = False


    def fit(self, x, y):
        x = np.atleast_2d(x)
        if x.shape[0] < self.min_sample:
            return
        self.is_fited = True
        y = np.asarray(y)
        if self.standardlize:
            self.Y_mean = y.mean()
            self.Y_std = y.std()
        else:
            self.Y_mean = 0
            self.Y_std = 1

        y = (y-self.Y_mean) / self.Y_std
        self.gpr = GPy.models.gp_regression.GPRegression(x, y, self.kernel)
        self.gpr.optimize(max_iters=100)
        # self.kernel = self.gpr.kern


    def predict(self, newX):
        assert self.is_fited
        return np.squeeze(self.gpr.predict(np.atleast_2d(newX))[0])*self.Y_std + self.Y_mean

    def cached_predict(self, newX):

        key = hash(newX.data.tobytes())
        if key in self.cached_mu_sigma:
            return self.cached_mu_sigma[key][0]
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, std = self.gpr.predict(np.atleast_2d(newX), full_cov=True)
            return np.squeeze(mu)*self.Y_std + self.Y_mean, np.squeeze(np.sqrt(std))*self.Y_std

    def cached_predict_with_sigma(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_sigma[key] = self.predict_with_sigma(newX)
        return self.cached_mu_sigma[key]

    def predict_with_cov(self, newX):
        assert self.is_fited
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, cov = self.gpr.predict(np.atleast_2d(newX), full_cov=True)
            return np.squeeze(mu)*self.Y_std + self.Y_mean, np.squeeze(cov)*self.Y_std**2

    def cached_predict_with_cov(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_cov[key] = self.predict_with_cov(newX)
        return self.cached_mu_cov[key]



class GaussianProcessRegressorARD_sklearn(Surrogate):
    def __init__(self, dim, min_sample=3):
        super(GaussianProcessRegressorARD_sklearn, self).__init__(dim, min_sample)
        self.cached = {}
        kernel = gaussian_process.kernels.ConstantKernel(
            constant_value=1#, constant_value_bounds=(1e-4, 1e4)
        ) * gaussian_process.kernels.RBF(
            length_scale=1#, length_scale_bounds=(1e-4, 1e4)
        )
        self.gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        self.is_fited = False

    def fit(self, x, y):
        x = np.atleast_2d(x)
        if x.shape[0] < self.min_sample:
            return
        self.gpr.fit(x, y)
        self.is_fited = True

    def predict(self, newX):
        assert self.is_fited
        return self.gpr.predict(np.atleast_2d(newX))

    def cached_predict(self, newX):

        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, std = self.gpr.predict(np.atleast_2d(newX), return_std=True)
            return mu, std


class GaussianProcessRegressor(Surrogate):
    def __init__(self, dim, min_sample=3):
        super().__init__(dim, min_sample)
        self.kernel = SEkernel()
        self.cached = {}
        self.cached_mu_sigma = {}
        self.cached_mu_cov = {}
        self.is_fited = False

    def fit(self, x, y):
        x = np.atleast_2d(x)
        if x.shape[0] < self.min_sample:
            return
        self.is_fited = True
        self.X = x
        kernel = self.kernel.compute_kernel(x)
        self.L = cholesky(kernel, lower=True)
        _part = solve_triangular(self.L, y, lower=True)
        self.KinvY = solve_triangular(self.L.T, _part, lower=False)

    def predict(self, newX):
        assert self.is_fited
        # Kstar = np.squeeze(self.kernel.compute_kernel(self.X, newX))
        Kstar = (self.kernel.compute_kernel(self.X, newX))
        return (Kstar.T @ self.KinvY).item()

    def cached_predict(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        assert self.is_fited
        if not hasattr(self, 'X'):
            return 0, np.inf
        else:
            Kstar = self.kernel.compute_kernel(self.X, newX)
            _LinvKstar = solve_triangular(self.L, Kstar, lower=True)
            return np.squeeze(Kstar.T @ self.KinvY), np.sqrt(self.kernel.compute_kernel(newX) - _LinvKstar.T @ _LinvKstar)

    def cached_predict_with_sigma(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_sigma[key] = self.predict_with_sigma(newX)
        return self.cached_mu_sigma[key]

    def cached_predict_with_cov(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_cov:
            self.cached_mu_cov[key] = self.predict_with_cov(newX)
        return self.cached_mu_cov[key]

    def predict_with_cov(self, newX):
        assert self.is_fited
        if not hasattr(self, 'X'):
            return 0, np.inf
        else:
            Kstar = self.kernel.compute_kernel(self.X, newX)
            _LinvKstar = solve_triangular(self.L, Kstar, lower=True)
            return np.squeeze(Kstar.T @ self.KinvY), (self.kernel.compute_kernel(newX) - _LinvKstar.T @ _LinvKstar)



class GaussianProcessRegressorARD_torch(Surrogate):
    def __init__(self, dim, min_sample=4, name='standard'):
        Surrogate.__init__(self,dim, min_sample)
        # self.cached = {}
        # self.cached_mu_sigma = {}
        # self.cached_mu_cov = {}


        self.is_fited = False
        assert name in ["standard", "gaussian"]
        mapping = {
            "standard": StandardTransform,
            "gaussian": GaussianTransform,
        }
        self.normalizer = mapping[name]
        # self.observed_z = torch.empty(size=(0, dim))
        self.y_observed = torch.empty(size=(0, 1))
        self.X_observed = torch.empty(size=(0, dim))

    def transform_outputs(self, y: np.array):
        # return y # TODO
        psi = self.normalizer(y)
        z = psi.transform(y)
        return z

    def fit(self, x, y):
        self.X_observed = torch.cat((self.X_observed, torch.Tensor(x)), dim=0)
        self.y_observed = torch.cat((self.y_observed, torch.Tensor(y).unsqueeze(1)), dim=0)
        # x = torch.atleast_2d(x)
        if self.X_observed.shape[-2] < self.min_sample:
            return
        self.is_fited = True

        # if y.ndim == 1:
        #     y = y[..., None]
        self.z_observed = torch.Tensor(self.transform_outputs(self.y_observed.cpu().numpy()))
        # self.gpr = SingleTaskGP(
        #     train_X=self.X_observed,
        #     train_Y=self.z_observed,
        #     # special likelihood for numerical Cholesky errors, following advice from
        #     # https://www.gitmemory.com/issue/pytorch/botorch/179/506276521
        #     # likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-3)),
        # )
        self.gpr = FixedNoiseGP(
            train_X=self.X_observed,
            train_Y=self.z_observed,
            train_Yvar=torch.full_like(self.z_observed, 1)
            # special likelihood for numerical Cholesky errors, following advice from
            # https://www.gitmemory.com/issue/pytorch/botorch/179/506276521
            # likelihood=GaussianLikelihood(noise_constraint=GreaterThan(1e-3)),
        )

        mll = ExactMarginalLogLikelihood(self.gpr.likelihood, self.gpr)
        # with gpytorch.settings.cholesky_jitter(1e-1):
        fit_gpytorch_model(mll)


    def get_posterior(self, newX):
        assert self.is_fited
        return self.gpr.posterior(torch.atleast_2d(newX))

