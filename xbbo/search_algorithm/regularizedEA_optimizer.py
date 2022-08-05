import math
from typing import Optional, List, Tuple, cast

import numpy as np
from xbbo.initial_design import ALL_avaliable_design

from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trial, Trials
from . import alg_register


@alg_register.register('rea')
class RegularizedEA(AbstractOptimizer):
    '''
    Regularized Evolution for Image Classifier Architecture Search
    '''
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 initial_design: str = 'random',
                 suggest_limit: int = np.inf,
                 init_budget: int = None,
                 sample_size=None,
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   suggest_limit=suggest_limit,
                                   seed=seed,
                                   **kwargs)
        # FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        # Uniform2Gaussian.__init__(self)
        # configs = self.space.get_hyperparameters()
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()
        # self.num_of_tour_particips = kwargs.get('n_part',2)
        self.init_budget = init_budget
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space,
            self.rng,
            ta_run_limit=suggest_limit,
            init_budget=init_budget,
            **kwargs)
        if self.init_budget is None:
            self.init_budget = self.initial_design.init_budget
        self.initial_design_configs = self.initial_design.select_configurations(
        )[:self.init_budget]
        self.pop_size = self.init_budget
        self.tournament_sample_size = self.pop_size // 2 if sample_size is None else min(
            max(sample_size, 1), self.pop_size)
        self.population_X = np.asarray([
            config.get_array(sparse=False)
            for config in self.initial_design_configs
        ])
        self.population_y = None

        # self.mu = kwargs.get('mu',20) # 交叉和变异算法的分布指数
        self.mum = kwargs.get('mum', 20)
        # self.crossrate = kwargs.get('crossrate',0.9)
        # ---

        self.trials = Trials(space, dim=self.dimension)
        self.cur = 0
        self.gen = 0
        self.listy = []

    def _suggest(self, n_suggestions=1):
        assert self.pop_size % n_suggestions == 0
        trial_list = []
        for n in range(n_suggestions):
            new_individual = self.population_X[self.cur]
            new_individual = np.clip(new_individual, self.bounds.lb,
                                     self.bounds.ub)
            # array = self.feature_to_array(new_individual)
            config = DenseConfiguration.from_array(self.space, new_individual)
            self.cur += 1
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=new_individual,
                      origin='REA',
                      loc=self.cur))

        # self._num_suggestions += n_suggestions
        return trial_list

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            self.listy.append(trial.observe_value)
        if self.cur == len(self.population_X):
            if self.population_y is None:
                self.population_y = np.asarray(self.listy)
            else:
                self.population_y = np.concatenate(
                    [self.population_y, self.listy], axis=0)

            # remove dead
            self.population_y = np.asarray(self.population_y[-self.pop_size:])
            self.population_X = np.asarray(self.population_X[-self.pop_size:])

            s_id = np.argsort(self.population_y)
            rank = np.argsort(s_id)
            parents_id = self.rng.choice(len(rank),
                                         replace=True,
                                         size=self.tournament_sample_size)
            parent_id = s_id[rank[parents_id].min()]

            children = self.__mutate2(self.population_X[parent_id])

            # ---
            self.population_X = np.append(self.population_X, [children],
                                          axis=0)
            self.cur = len(self.population_y)
            assert self.cur < len(self.population_X)

            self.listy = []
            self.gen += 1

    def calculate_crowding_distance(self, points, m, M):
        cd = np.zeros(len(points))
        if len(cd) <= points.shape[-1]:
            return cd
        for n in range(points.shape[-1]):
            s_id = np.argsort(points[:, n])
            cd[s_id[[0, -1]]] = [np.inf, np.inf]
            for idx in range(len(s_id[1:-1])):
                cd[s_id[idx]] += (points[s_id[idx + 1], n] -
                                  points[s_id[idx - 1], n]) / (M[n] - m[n])
        return cd

    def __mutate2(self, parent):
        child = parent.copy()
        r = self.rng.rand(len(child))
        delta = np.where(r < 0.5, (2 * r)**(1 / (self.mum + 1)) - 1,
                         1 - (2 * (1 - r))**(1 / (self.mum + 1)))
        return np.clip(child + delta, self.bounds.lb, self.bounds.ub)

    def __mutate_naive(self, parent):
        child = parent.copy()
        child[self.rng.randint(self.dimension)] = self.rng.uniform(0, 1)
        return child

    def __mutate(self, parent, mu0=0.7):
        if self.rng.rand() <= mu0:
            return self.__mutate_naive(parent)
        else:
            return self.rng.uniform(0, 1, size=self.dimension)


opt_class = RegularizedEA
