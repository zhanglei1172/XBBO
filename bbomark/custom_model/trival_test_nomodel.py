from scipy.optimize import minimize
import numpy as np

from bbomark.core import TestFunction, AbstractBaseModel
from bbomark.constants import VISIBLE_TO_OPT

class Model(TestFunction):
    objective_names = (VISIBLE_TO_OPT, "generalization")
    def __init__(self, model, dataset, metric, shuffle_seed=0, data_root=None):

        super().__init__()
        self.a = 1
        self.b = 100
        self.api_config = self._load_api_config()


    def evaluate(self, params):
        """
        Rosenbrock function
        """
        func_val = 0
        for k in sorted(params.keys())[::2]:
            x = params[k.rsplit('_', maxsplit=1)[0]+'_0']
            y = params[k.rsplit('_', maxsplit=1)[0] + '_1']
            func_val += (self.a-x)**2 + self.b*(y-x**2)**2
        return func_val + 0.1*np.random.randn()*func_val, 0

    def _load_api_config(self):
        return {
            'x_0': {'type': 'real', 'range': (-10, 10)},
            'x_1': {'type': 'real', 'range': (-10, 10)}
        }