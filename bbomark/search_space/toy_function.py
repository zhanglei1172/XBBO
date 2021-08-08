import numpy as np

from bbomark.core import TestFunction

class Model(TestFunction):
    def __init__(self, cfg, **kwargs):
        # np.random.seed(cfg.GENERAL.random_seed)
        self.cfg = cfg
        self.dim = 30
        # assert self.dim % 2 == 0
        super().__init__()


        assert cfg.TEST_PROBLEM.kwargs.func_name in ('rosenbrock')
        # func_name = cfg.TEST_PROBLEM.kwargs.func_name
        func_name = kwargs.get('func_name')
        if func_name == 'rosenbrock':
            self.func = Rosenbrock(self.cfg.TEST_PROBLEM.SEARCH_SPACE.hp.cat_dim)
        self.noise_std = kwargs.get('noise_std')
        self.api_config = self._load_api_config()

    def evaluate(self, params: dict):

        input_x = []
        for k in sorted(params.keys()):
            input_x.append(params[k])
        f = self.func(np.asarray(input_x))
        random_noise = np.random.randn() * self.noise_std + 1.
        res_out = {
            'row': f,
            'noise': f*random_noise,
        }
        res_loss = {
            'test': f,
            'val': f * random_noise,
        }
        return (
            [res_out[k] for k in self.cfg.TEST_PROBLEM.func_evals],
            [res_loss[k] for k in self.cfg.TEST_PROBLEM.losses]
        )

    def _load_api_config(self):
        return self.func._load_api_config()



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
            f_x += 100 * (input_x[i + 1] - input_x[i] ** 2) ** 2 + (1 - input_x[i]) ** 2


        return f_x

    def _load_api_config(self):
        return {
            'x_{}'.format(k): {'type': 'float', 'range': (-10, 10)} for k in range(self.dim)
        }