# from scipy.optimize import minimize

import numpy as np

from bbomark.constants import VISIBLE_TO_OPT
from bbomark.core import TestFunction, AbstractBaseModel


class Model(TestFunction):
    objective_names = (VISIBLE_TO_OPT, "generalization")
    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):

        super().__init__()
        self.a = 1
        self.b = 100
        state = np.random.RandomState(2021)
        self.X_init = state.uniform(low=-10, high=10, size=(10, 2))
        self.api_config = self._load_api_config()
        # self.grad = lambda x, y: np.array([-2*(self.a-x) - 4*self.b*(y-x**2), 2*self.b*(y-x**2)])
        self.grad = lambda x, y: np.array([2*x, 2*y])

    def rosen(self, X):
        x = X[0]
        y = X[1]
        func_val = 0
        func_val += (self.a-x)**2 + self.b*(y-x**2)**2
        return func_val+np.random.rand()

    def func(self, x):
        return np.sum(x**2) + np.random.rand()

    def evaluate(self, params):
        """
        Rosenbrock function
        """
        # res = minimize(self.rosen, self.X_init)
        x_init = np.array([params['x_0'], params['x_1']])
        res = self.sgd(x_init)

        return res, 0

    def _load_api_config(self):
        return {
            'x_0': {'type': 'real', 'range': (-10, 10)},
            'x_1': {'type': 'real', 'range': (-10, 10)}
        }

    def sgd(self, x_init):
        iter_num = 100
        eta = 0.01
        for iter in range(iter_num):
            x_init_ = x_init - eta*(self.grad(x_init[0], x_init[1])+np.random.rand())
            if np.allclose(x_init, x_init_):
                break
            x_init = x_init_
        # v = self.rosen(x_init)
        v = self.func(x_init)

        return 10e8 if np.isnan(v) else v