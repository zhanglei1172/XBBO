import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import tqdm, random
# from sklearn.ensemble import RandomForestClassifier
# from scipy.optimize import minimize
import torch
import torch.nn as nn
from torch.optim import Adam
from xbbo.core import AbstractOptimizer
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         CategoricalHyperparameter,
                                         OrdinalHyperparameter)
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration


class FloatVarKDE():
    def __init__(self, low=0, high=1, bandwidth=0.1):
        self.bandwidth = bandwidth
        # self.kde = NaiveKDE(bandwidth=bandwidth)
        self.kde = KernelDensity(bandwidth=bandwidth)
        self.low = low
        self.high = high
        # self.kde.fit(high-low)

    def sample(self, num):
        cnt = 0
        samples = []
        while cnt < num:
            sample = self.kde.sample().item()
            if self.low <= sample <= self.high:
                cnt += 1
                samples.append(sample)
        return np.asarray(samples)

    def fit(self, points):
        self.kde.fit(points.reshape(-1, 1))

    def log_pdf(self, x):
        return self.kde.score_samples(x.reshape(-1, 1))


class Category():
    '''
    every point has a instance of this class
    '''
    def __init__(self, choice, dim, top=0.9):
        '''
        choice为这个point的选择
        '''
        self.choice = choice
        self.dim = dim
        self.prob = self._get_prob()
        self.top = top

    def _get_prob(self):
        p = []
        for i in range(self.dim):
            if self.choice == i:
                p.append(self.top)
            else:
                p.append((1 - self.top) / (self.dim - 1))
        return p

    def sample(self, ):
        return np.random.multinomial(1, self.prob, size=1)

    def log_pdf(self, x):
        return np.log(self.prob[x])


class Category_PE():
    '''
    all good/bad points have the same instance of this class
    '''
    def __init__(self, choices_idx):
        self.dim = len(choices_idx)
        self.choices_idx = choices_idx

    def sample(self, num):
        samples = []
        pt_idx = np.random.multinomial(1, self.prob, size=num)
        for idx in pt_idx:
            samples.append(self.point_kernel_model[idx].sample())
        return samples

    def log_pdf(self, x):
        return sum(m.log_pdf(x) for m in self.point_kernel_model)

    def fit(self, points):
        self.prob = np.zeros(len(points))
        self.point_kernel_model = [
            Category(point, self.dim) for point in points
        ]


class Toy_TPE(AbstractOptimizer):
    # @kdeBackend(FloatVarKDE)
    def __init__(self,
                 config_spaces,
                 bandwidth=1,
                 min_sample=30,
                 gamma=0.1,
                 candidates_num=24):
        AbstractOptimizer.__init__(self, config_spaces)
        self.gamma = gamma
        self.min_sample = min_sample
        self.bandwidth = bandwidth
        self.candidates_num = candidates_num
        configs = self.space.get_hyperparameters()
        self.lx = []
        self.gx = []
        for config in configs:
            if isinstance(config, CategoricalHyperparameter):
                self.lx.append(Category_PE(config.num_choices))
                self.gx.append(Category_PE(config.num_choices))
            elif isinstance(config, OrdinalHyperparameter):
                self.lx.append(Category_PE(config.num_elements))
                self.gx.append(Category_PE(config.num_elements))
            else:
                self.lx.append(FloatVarKDE())
                self.gx.append(FloatVarKDE())
        self.hp_num = len(configs)
        self.history = []
        self.history_y = []

    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if len(self.history_y) <= self.min_sample or np.random.rand() < 0.1:
            return self._random_suggest()
        else:
            sas = []
            for n in range(n_suggestions):
                suggest_array = []
                for i in range(self.hp_num):
                    candidates = self.lx[i].sample(self.candidates_num)
                    lx = self.lx[i].log_pdf(candidates)
                    gx = self.gx[i].log_pdf(candidates)
                    max_idx = np.argmax(lx - gx)
                    suggest_array.append(candidates[max_idx])
                sas.append(suggest_array)
        x = [DenseConfiguration.array_to_dict(self.space,
                                             np.array(sa)) for sa in sas]
        return x, sas

    def _random_suggest(self, n_suggestions=1):
        sas = []
        for n in range(n_suggestions):
            suggest_array = []
            for config in self.space.get_hyperparameters():
                if isinstance(config, CategoricalHyperparameter):
                    suggest_array.append(np.random.choice(config.num_choices))
                elif isinstance(config, OrdinalHyperparameter):
                    suggest_array.append(np.random.choice(config.num_elements))
                else:
                    suggest_array.append(np.random.rand())
            sas.append(suggest_array)
        x = [DenseConfiguration.array_to_dict(self.space,
                                             np.array(sa)) for sa in sas]
        return x, sas

    def _split_l_g(self):
        samples = np.asarray(self.history)
        good_num = int(self.gamma * len(samples))
        samples_sorted = samples[np.argsort(self.history_y).ravel(),...]
        good_points = samples_sorted[:good_num, :]
        bad_points = samples_sorted[good_num:, :]
        return good_points, bad_points

    def observe(self, x, y):
        for n_suggest in range(len(x)):
            self.history.append(x[n_suggest])
            self.history_y.append(y[n_suggest])
        if len(self.history_y) < self.min_sample:
            return

        good_points, bad_points = self._split_l_g()
        for i in range(self.hp_num):
            self.lx[i].fit(good_points[:, i])
            self.gx[i].fit(bad_points[:, i])


opt_class = Toy_TPE