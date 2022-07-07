from typing import List
import numpy as np
from xbbo.initial_design import ALL_avaliable_design

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from . import alg_register


@alg_register.register('pso')
class PSO(AbstractOptimizer):
    '''
    ref: https://github.com/guofei9987/scikit-opt
    '''
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 init_budget=40,
                 suggest_limit=np.inf,
                 initial_design='sobol',
                 w=0.8,
                 c1=0.5,
                 c2=0.5,
                 boundary_fix_type='random',
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   **kwargs)

        # Uniform2Gaussian.__init__(self,)
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()

        init_budget = max(init_budget, 40)
        self.fix_type = boundary_fix_type
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space,
            self.rng,
            ta_run_limit=suggest_limit,
            init_budget=init_budget,
            **kwargs)
        self.init_budget = init_budget
        if self.init_budget is None:
            self.init_budget = self.initial_design.init_budget
        self.pop_size = self.init_budget
        self.trials = Trials(space, dim=self.dimension)

        self.w = w  # inertia
        self.cp, self.cg = c1, c2  # parameters to control personal best, global best respectively
        self.cur = 0
        self.gen = 0
        self.initial_design_configs = self.initial_design.select_configurations(
        )[:self.init_budget]
        self.population_X = np.asarray([
            config.get_array(sparse=False)
            for config in self.initial_design_configs
        ])
        # self.population_X = self.rng.uniform(low=self.bounds.lb,
        #                                      high=self.bounds.ub,
        #                                      size=(self.pop_size,
        #                                            self.dimension))
        v_high = self.bounds.ub - self.bounds.lb
        self.V = self.rng.uniform(low=-v_high,
                                  high=v_high,
                                  size=(self.pop_size,
                                        self.dimension))  # speed of particles
        self.pbest_x = self.population_X.copy(
        )  # personal best location of every particle in history
        self.pbest_y = np.array(
            [np.inf] *
            self.pop_size)  # best image of every particle in history
        self.gbest_x = self.pbest_x.mean(axis=0).reshape(
            1, -1)  # global best location for all particles
        self.gbest_y = np.inf  # global best y for all particles
        self.gbest_y_hist = []  # gbest_y of every iteration
        self.listy = []

    def fix_boundary(self, individual):
        if self.fix_type == 'random':
            return np.where(
                (individual > self.bounds.lb) & (individual < self.bounds.ub),
                individual,
                self.rng.uniform(self.bounds.lb, self.bounds.ub,
                                 self.dimension))  # FIXME
        elif self.fix_type == 'clip':
            return np.clip(individual, self.bounds.lb, self.bounds.ub)

    def _suggest(self, n_suggestions=1):
        assert self.pop_size % n_suggestions == 0
        trial_list = []
        for n in range(n_suggestions):
            new_individual = self.population_X[self.cur]
            # new_individual = np.clip(new_individual, self.bounds.lb, self.bounds.ub)
            # self.fix_boundary(new_individual)
            # array = self.feature_to_array(new_individual)
            config = DenseConfiguration.from_array(self.space, new_individual)
            self.cur += 1
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=new_individual,
                      origin='PSO',
                      loc=self.cur))

        # self._num_suggestions += n_suggestions
        return trial_list

    def _observe(self, trial_list: List[Trial]):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            self.listy.append(trial.observe_value)
        if self.cur == len(self.population_X):

            self.population_y = np.asarray(self.listy)

            self.update_pbest()
            self.update_gbest()
            self.update_V()
            self.update_X()

            self.cur = 0

            self.listy = []
            self.gen += 1
            self.gbest_y_hist.append(self.gbest_y)

    def update_V(self):
        r1 = self.rng.rand(self.pop_size, self.dimension)
        r2 = self.rng.rand(self.pop_size, self.dimension)
        self.V = self.w * self.V + \
                 self.cp * r1 * (self.pbest_x - self.population_X) + \
                 self.cg * r2 * (self.gbest_x - self.population_X)

    def update_X(self):
        self.population_X = self.population_X + self.V
        self.population_X = self.fix_boundary(self.population_X)
        # self.population_X = np.clip(self.population_X, self.lb, self.ub)

    def update_pbest(self):
        '''
        personal best
        :return:
        '''
        self.need_update = self.pbest_y > self.population_y
        # for idx, x in enumerate(self.population_X):
        #     if self.need_update[idx]:
        #         self.need_update[idx] = self.check_constraint(x)

        self.pbest_x = np.where(np.expand_dims(self.need_update, axis=-1),
                                self.population_X, self.pbest_x)
        self.pbest_y = np.where(self.need_update, self.population_y,
                                self.pbest_y)

    def update_gbest(self):
        '''
        global best
        :return:
        '''
        idx_min = self.pbest_y.argmin()
        if self.gbest_y > self.pbest_y[idx_min]:
            self.gbest_x = self.population_X[idx_min, :].copy()
            self.gbest_y = self.pbest_y[idx_min]


opt_class = PSO
