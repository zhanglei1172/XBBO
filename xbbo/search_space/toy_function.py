import numpy as np

from . import problem_register
from xbbo.core.abstract_model import TestFunction

@problem_register.register('toy-problems')
class Model(TestFunction):
    _SUPPORT_FUNCTIONS = ('rosenbrock', 'rastrigin', 'indexsum', 'toyrebar',
                          'branin', 'ZDT1')

    def __init__(self, seed, **kwargs):
        # np.random.seed(cfg.GENERAL.random_seed)
        # self.dim = 30
        # assert self.dim % 2 == 0
        super().__init__(seed=seed)

        assert kwargs["func_name"] in self._SUPPORT_FUNCTIONS
        # func_name = cfg.TEST_PROBLEM.kwargs.func_name
        func_name = kwargs.get('func_name')
        if func_name == 'rosenbrock':
            self.func = Rosenbrock(
                kwargs.get("dim"))
        elif func_name == 'rastrigin':
            self.func = Rastrigin(
                kwargs.get("dim"))
        elif func_name == 'indexsum':
            self.func = IndexSum(kwargs.get("dim"))
        elif func_name == 'toyrebar':
            self.func = ToyREBAR(kwargs.get("dim"),
                                 kwargs.get("t"))
        elif func_name == 'branin':
            self.func = Branin()
        elif func_name == 'ZDT1':
            self.func = ZDT1()
        else:
            assert False
        self.noise_std = kwargs.get('noise_std', 0)
        self.api_config = self._load_api_config()

    def evaluate(self, params: dict):

        input_x = []
        for k in sorted(params.keys()):
            input_x.append(params[k])
        f = self.func(np.asarray(input_x))
        if isinstance(f, tuple):
            random_noise = self.rng.randn(len(f)) * self.noise_std + 1.
        else:
            random_noise = self.rng.randn() * self.noise_std + 1.
        # res_out = {
        #     # 'raw': f,
        #     'noise': f * random_noise,
        # }
        # res_loss = {
        #     'test': f,
        #     'val': f * random_noise,
        # }
        # return (res_out['noise'])
        return f * random_noise

    def _load_api_config(self):
        return self.func._load_api_config()


class Rastrigin():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input_x):
        f_x = 10. * input_x.shape[0]
        for i in input_x:
            f_x += i**2 - 10 * np.cos(2 * np.pi * i)
        return f_x

    def _load_api_config(self):
        return {
            'x_{}'.format(k): {
                'type': 'float',
                'range': (-5.12, 5.12)
            }
            for k in range(self.dim)
        }


class IndexSum():
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, input_x):
        f_x = 0
        for i in input_x:
            f_x += i  # cat都取第一个类最优
        return f_x

    def _load_api_config(self):
        return {
            'x_{}'.format(k): {
                'type': 'cat',
                'values': list(range(10))
            }
            for k in range(self.dim)
        }


class ToyREBAR():
    # min
    def __init__(self, dim, t=0.45):
        # self.a = 1
        # self.b = 100
        self.dim = dim
        self.t = t
        pass

    def __call__(self, input_x):
        """
        Rosenbrock function
        """
        f_x = 0
        for i in input_x:
            f_x += (i - self.t)**2

        return f_x / input_x.shape[0]

    # def _load_api_config(self):
    #     return {
    #         'x_{}'.format(k): {'type': 'cat', 'values': list(range(-5,6))} for k in range(self.dim)
    #     }

    def _load_api_config(self):
        '''
        参数是b的value
        '''
        return {
            'b_{}'.format(k): {
                'type': 'cat',
                'values': (0, 1)
            }
            for k in range(self.dim)
        }


class Rosenbrock():
    # min
    def __init__(self, dim):
        # self.a = 1
        # self.b = 100
        self.dim = dim
        pass

    def __call__(self, input_x):
        """
        Rosenbrock function
        """
        f_x = 0
        for i in range(input_x.shape[0] - 1):
            f_x += 100 * (input_x[i + 1] - input_x[i]**2)**2 + (1 -
                                                                input_x[i])**2

        return f_x

    # def _load_api_config(self):
    #     return {
    #         'x_{}'.format(k): {'type': 'cat', 'values': list(range(-5,6))} for k in range(self.dim)
    #     }

    def _load_api_config(self):
        return {
            'x_{}'.format(k): {
                'type': 'float',
                'range': (-5, 10)
            }
            for k in range(self.dim)
        }


class Branin():
    '''
    The Branin, or Branin-Hoo, function has three global minima. The recommended values of a, b, c, r, s and t are: a = 1, b = 5.1 ⁄ (4π2), c = 5 ⁄ π, r = 6, s = 10 and t = 1 ⁄ (8π).
    https://www.sfu.ca/~ssurjano/branin.html
    global minimum:
    f(x*) = 0.397887
    x* = (-pi, 12.275)、(pi, 2.275)、(9.42478, 2.475)
    '''
    def __init__(self, ):
        self.a = 1
        self.b = 5.1 / (4 * np.pi**2)
        self.c = 5 / np.pi
        self.r = 6
        self.s = 10
        self.t = 1 / (8 * np.pi)

    def __call__(self, input_x):
        x1, x2 = input_x
        term1 = self.a * (x2 - self.b * x1**2 + self.c * x1 - self.r)**2
        term2 = self.s * (1 - self.t) * np.cos(x1)

        y = term1 + term2 + self.s
        return y

    def _load_api_config(self):
        return {
            "x1": {
                "type": "float",
                "range": [-5, 10]
            },
            "x2": {
                "type": "float",
                "range": [0, 15]
            }
        }


class ZDT1():

    def __init__(self, ):
        self.n = 30

    def __call__(self, input_x):
        sigma = sum(input_x[1:])
        g = 1 + sigma * 9 / (self.n - 1)
        h = 1 - (input_x[0] / g)**0.5
        return input_x[0], g*h

    def _load_api_config(self):
        return {
            "x1": {
                "type": "float",
                "range": [0, 1]
            },
            "x2": {
                "type": "float",
                "range": [0, 1]
            }
        }
