import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Record:

    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        # self.n_calls = cfg.OPTM.max_call
        self.features = [] # n_call * n_suggest * dim # 单独存储
        self.func_evals = [] # n_call * n_suggest * dim
        # self.losses = [] # n_call * n_suggest * 2
        # self.cum_min = {
        #     'cum_min_suggest_dict': [],
        #     'cum_min_losses': [],
        #     'cum_min_func_evals': []
        # }
        self.current_best = np.inf
        # self.metrics = [] # n_call * n_suggest * dim
        # self.budgets = []
        self.timing = { # 用来评估优化器的性能，如果是把model推理时间作为评估，应该放在loss、func_evals中
            'suggest_time_per_suggest': [],
            'observe_time_per_suggest': [],
            'eval_time_per_suggest': []
        }
        self.suggest_dict = []
        # index = pd.MultiIndex.from_product([self.n_calls * n_suggest, dim])
        self.df = None

    # def __str__(self):
    #     return np.asarray(self.func_evals).__str__()

    def size(self):
        return len(self.func_evals)

    def append(self, feature, y, suggest_point, timing):
        self.features.append(feature)
        self.func_evals.append(y)
        # self.losses.append(loss)
        for k in self.timing:
            self.timing[k].append(timing[k])
        self.suggest_dict.append(suggest_point)

    # def get_best(self):
    #     idx = np.argmin(np.asarray(self.func_evals)[...,0].ravel())
    #     return idx//len(self.func_evals[0]), \
    #            np.asarray(self.features).ravel()[idx],\
    #            np.asarray(self.func_evals)[...,0].ravel()[idx],\
    #            np.asarray(self.suggest_dict).ravel()[idx]

    # def plot(self):
    #     plt.plot(self.history)

    def is_duplicate(self, x, rtol=1e-5, atol=1e-8):
        return any(np.allclose(x_prev, x, rtol=rtol, atol=atol)
                   for x_prev in self.features)

    def save_to_file(self, r):
        if not os.path.exists(self.exp_dir):
            assert False
            # os.mkdir(cfg.GENARAL.exp_dir)
        if not os.path.exists(self.exp_dir + '/res'):
            os.mkdir(self.exp_dir + '/res/')
        self.func_evals = np.asarray(self.func_evals)
        # self.losses = np.asarray(self.losses)
        self.suggest_dict = np.asarray(self.suggest_dict)
        # reform = {
        #     (call, suggest): self.losses[call, suggest]
        #     for call in range(self.losses.shape[0])
        #     for suggest in range(self.losses.shape[1])
        # }
        self.df = pd.DataFrame(self.func_evals)#.T
        # self.losses_df.columns = pd.MultiIndex.from_product([['loss'], self.cfg.TEST_PROBLEM.losses])
        self.df.columns = ['func_evals']
        self.df['cum_func_evals'] = self.df['func_evals'].cummin()
        # self.losses_df.columns = cfg.TEST_PROBLEM.losses
        # self.losses_df.index.set_names(['call', 'suggest'], inplace=True)

        # reform = {
        #     (call, suggest): self.func_evals[call, suggest, :]
        #     for call in range(self.func_evals.shape[0])
        #     for suggest in range(self.func_evals.shape[1])
        # }
        # self.func_evals_df = pd.DataFrame(reform).T
        # self.func_evals_df.columns = pd.MultiIndex.from_product([['func_evals'], self.cfg.TEST_PROBLEM.func_evals])
        # # self.func_evals_df.columns = cfg.TEST_PROBLEM.func_evals
        # self.func_evals_df.index.set_names(['call', 'suggest'], inplace=True)

        # sd = pd.DataFrame(self.suggest_dict).stack()
        # sd.index.set_names(['call', 'suggest'], inplace=True)
        # sd.name = 'suggest_dict'
        # self.df = pd.concat([self.losses_df, self.func_evals_df], axis=1)
        # self.df['suggest_dict'] = sd

        self.time_df = pd.DataFrame(self.timing)
        self.time_df.index.set_names(['call'], inplace=True)

        # self.cum_min_df = pd.DataFrame(self.cum_min)
        # self.cum_min_df.index.set_names(['call'], inplace=True)
        self.df.to_csv(self.exp_dir + f'/res/res_{r}.csv')
        self.time_df.to_csv(self.exp_dir + f'/res/time_{r}.csv')
        # self.cum_min_df.to_csv(self.exp_dir + f'/res/cum_min_{r}.csv')

        self.features = np.asarray(self.features)
        np.savez(self.exp_dir + f'/res/array_{r}.npz', features=self.features)

        # if cfg.TEST_PROBLEM.metrics:
        #     reform = {
        #         (call, suggest): self.metrics[call, suggest, :]
        #         for call in range(self.metrics.shape[0])
        #         for suggest in range(self.metrics.shape[1])
        #     }
        #     self.metrics = pd.DataFrame(reform).T
        #     self.metrics.columns = cfg.TEST_PROBLEM.metrics
        # index = pd.MultiIndex.from_product([
        #     [f'call_{i}' for i in range(self.losses.shape[0])],
        #     [f'suggest_{i}' for i in range(self.losses.shape[1])]
        # ]
        # )