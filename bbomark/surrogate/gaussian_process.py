from scipy.linalg import solve_triangular, cholesky
from scipy import stats
import numpy as np

from bbomark.surrogate.base import Surrogate


class SEkernel():
    def __init__(self):
        self.initialize()

    def initialize(self):
        # self.sumF = 0.001
        # self.sumL = 0.001
        # self.sumY = 0.001
        self.sigma_f = 1
        self.sigma_l = 1
        self.sigma_y = 0.001



    def compute_kernel(self, x1, x2=None):
        if x2 is None:
            x2 = x1
            # noise = np.diag([self.sigma_y**2 for _ in range(x1.shape[0])])
            noise = np.eye(x1.shape[0]) * self.sigma_y**2
        else:
            noise = 0
        x2 = np.atleast_2d(x2)
        x1 = np.atleast_2d(x1)
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * (x1 @ x2.T)
        return self.sigma_f ** 2 * np.exp(-0.5 / self.sigma_l ** 2 * dist_matrix) + noise





class GaussianProcessRegressor(Surrogate):
    def __init__(self, dim):
        super().__init__(dim)
        self.kernel = SEkernel()
        self.cached = {}

    def fit(self, x, y):
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
            return (Kstar.T @ self.KinvY).item(), np.sqrt(self.kernel.compute_kernel(newX, newX) - _LinvKstar.T @ _LinvKstar)
