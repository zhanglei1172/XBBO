from scipy.linalg import solve_triangular, cholesky
from scipy import stats
import numpy as np
import GPy
from sklearn import gaussian_process

from bbomark.surrogate.base import Surrogate


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
    def __init__(self, dim):
        super(GaussianProcessRegressorARD_gpy, self).__init__(dim)
        self.cached = {}
        self.cached_mu_sigma = {}
        self.cached_mu_cov = {}
        self.kernel = GPy.kern.RBF(input_dim=self.dim,
                              variance=0.001,
                              lengthscale=0.5,
                              ARD=True)

        self.is_fited = False


    def fit(self, x, y):
        self.is_fited = True

        self.gpr = GPy.models.gp_regression.GPRegression(x, y[:, None],kernel=self.kernel)
        self.gpr.optimize(max_iters=100)

    def predict(self, newX):
        return np.squeeze(self.gpr.predict(np.atleast_2d(newX))[0])

    def cached_predict(self, newX):

        key = hash(newX.data.tobytes())
        if key in self.cached_mu_sigma:
            return self.cached_mu_sigma[key][0]
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, std = self.gpr.predict(np.atleast_2d(newX), full_cov=True)
            return np.squeeze(mu), np.squeeze(np.sqrt(std))

    def cached_predict_with_sigma(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_sigma[key] = self.predict_with_sigma(newX)
        return self.cached_mu_sigma[key]

    def predict_with_cov(self, newX):
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, cov = self.gpr.predict(np.atleast_2d(newX), full_cov=True)
            return np.squeeze(mu), np.squeeze(cov)

    def cached_predict_with_cov(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached_mu_sigma:
            self.cached_mu_cov[key] = self.predict_with_cov(newX)
        return self.cached_mu_cov[key]


class GaussianProcessRegressorARD_sklearn(Surrogate):
    def __init__(self, dim):
        super(GaussianProcessRegressorARD_sklearn, self).__init__(dim)
        self.cached = {}
        kernel = gaussian_process.kernels.ConstantKernel(
            constant_value=1#, constant_value_bounds=(1e-4, 1e4)
        ) * gaussian_process.kernels.RBF(
            length_scale=1#, length_scale_bounds=(1e-4, 1e4)
        )
        self.gpr = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=2)
        self.is_fited = False

    def fit(self, x, y):
        self.gpr.fit(x, y)
        self.is_fited = True

    def predict(self, newX):
        return self.gpr.predict(np.atleast_2d(newX))

    def cached_predict(self, newX):

        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
        if not self.is_fited:
            return 0, np.inf
        else:
            mu, std = self.gpr.predict(np.atleast_2d(newX), return_std=True)
            return mu, std


class GaussianProcessRegressor(Surrogate):
    def __init__(self, dim):
        super().__init__(dim)
        self.kernel = SEkernel()
        self.cached = {}
        self.cached_mu_sigma = {}
        self.cached_mu_cov = {}
        self.is_fited = False

    def fit(self, x, y):
        self.is_fited = True
        self.X = x
        kernel = self.kernel.compute_kernel(x)
        self.L = cholesky(kernel, lower=True)
        _part = solve_triangular(self.L, y, lower=True)
        self.KinvY = solve_triangular(self.L.T, _part, lower=False)

    def predict(self, newX):
        # Kstar = np.squeeze(self.kernel.compute_kernel(self.X, newX))
        Kstar = (self.kernel.compute_kernel(self.X, newX))
        return (Kstar.T @ self.KinvY).item()

    def cached_predict(self, newX):
        key = hash(newX.data.tobytes())
        if key not in self.cached:
            self.cached[key] = self.predict(newX)
        return self.cached[key]

    def predict_with_sigma(self, newX):
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
        if not hasattr(self, 'X'):
            return 0, np.inf
        else:
            Kstar = self.kernel.compute_kernel(self.X, newX)
            _LinvKstar = solve_triangular(self.L, Kstar, lower=True)
            return np.squeeze(Kstar.T @ self.KinvY), (self.kernel.compute_kernel(newX) - _LinvKstar.T @ _LinvKstar)
