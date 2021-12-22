import math

import numpy as np
import torch
from botorch.utils import draw_sobol_samples

from xbbo.core import TestFunction

NUM_BASE_TASKS = 5
from botorch.utils.transforms import normalize, unnormalize


def task_shift(task):
    """
    Fetch shift amount for task.
    """
    return math.pi * task /12.0
# set shift for target task

TARGET_SHIFT = 0.0
BOUNDS = torch.tensor([[-10.0], [10.0]])

def f(X, shift=TARGET_SHIFT):
    """
    Torch-compatible objective function for the target_task
    """
    f_X = X * torch.sin(X + math.pi + shift) + X/10.0
    return -f_X

def create_task():
    noise_std = 0.05

    # Sample data for each base task
    data = []
    data_y = []
    for task in range(NUM_BASE_TASKS):
        num_training_points = 50
        # draw points from a sobol sequence
        raw_x = draw_sobol_samples(
            bounds=BOUNDS, n=num_training_points, q=1, seed=task + 5397923,
        ).squeeze(1)
        # get observed values
        f_x = f(raw_x, task_shift(task + 1))
        train_y = f_x + noise_std * torch.randn_like(f_x)
        # train_yvar = torch.full_like(train_y, noise_std ** 2)
        data.append(normalize(raw_x, bounds=BOUNDS).numpy())
        data_y.append(train_y.numpy())
    return data, data_y
        # # store training data
        # data_by_task[task] = {
        #     # scale x to [0, 1]
        #     'train_x': normalize(raw_x, bounds=BOUNDS),
        #     'train_y': train_y,
        #     'train_yvar': train_yvar,
        # }

class Model(TestFunction):

    def __init__(self, cfg, **kwargs):
        # np.random.seed(cfg.GENERAL.random_seed)
        self.cfg = cfg
        # self.dim = 30
        # assert self.dim % 2 == 0
        super().__init__()
        # name = kwargs.get('func_name', None)
        self.hp_names = ['hp_0']
        self.noise_std = kwargs.get('noise_std', 0)
        self.api_config = self._load_api_config()
        self.old_D_x, self.old_D_y = create_task()
        # self.best_err = min(self.new_D_y).item()
        # self.err_range = max(self.new_D_y).item() - self.best_err
        # self.sorted_new_D_y = np.sort(self.new_D_y).ravel()
        self.new_D_x = None

    def evaluate(self, params: dict):

        # input_x = []

        f_v = f(unnormalize(torch.Tensor([params[k] for k in self.hp_names]), BOUNDS), TARGET_SHIFT).numpy()
        if self.noise_std == 0:
            random_noise = 1
        else:
            random_noise = np.random.randn() * self.noise_std + 1.
        # regret = (f - self.best_err) / self.err_range
        res_out = {
            # 'rank': (np.searchsorted(self.func.sorted_new_D_y, -f)+1)/len(self.func.sorted_new_D_y),
            # 'rank': (np.searchsorted(self.sorted_new_D_y, f) + 1) / len(self.sorted_new_D_y),
            # 'regret': regret,
            # 'log_regret': np.log10(regret)
            'f_v': f_v,
        }
        res_loss = {
            'test': f_v,
            'val': f_v * random_noise,
        }
        return ([res_out[k] for k in self.cfg.TEST_PROBLEM.func_evals],
                [res_loss[k] for k in self.cfg.TEST_PROBLEM.losses])

    def _load_api_config(self):
        return {
            hp_name: {
                'type': 'float', 'range': [0, 1]
            } for hp_name in self.hp_names
        }

    def _inst_to_config(self, inst):
        hp_params = {}
        for i, hp in enumerate(self.api_config):
            hp_params[hp] = inst[i]
        return hp_params

    def array_to_config(self, ret_param=True):
        if ret_param:
            self.old_D_x_params = []
            for d in range(len(self.old_D_x)):
                self.old_D_x_params.append([self._inst_to_config(inst) for inst in self.old_D_x[d]])

            self.new_D_x_param = [self._inst_to_config(inst) for inst in self.new_D_x]

            return self.old_D_x_params, self.old_D_y, self.new_D_x_param
        else:
            return self.old_D_x, self.old_D_y, self.new_D_x

