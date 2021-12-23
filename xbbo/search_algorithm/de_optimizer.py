import numpy as np

from xbbo.configspace.feature_space import FeatureSpace_gaussian
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration
from xbbo.core.trials import Trials


class DE(AbstractOptimizer, FeatureSpace_gaussian):

    def __init__(self,
                 config_spaces,
                 llambda=10):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        configs = self.space.get_hyperparameters()
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)

        self.hp_num = len(configs)
        self.llambda = llambda
        self.population = [None] * self.llambda
        self.population_fitness = [None] * self.llambda
        self.candidates = [None] * self.llambda

        self.current_best = None
        self.current_best_fitness = np.inf
        self._num_suggestions = 0
        self.F1 = 0.8
        self.F2 = 0.8
        self.CR = 0.5
        self.trials = Trials()

    def suggest(self, n_suggestions=1):

        sas = []
        x_arrays = []
        for n in range(n_suggestions):
            # suggest_array = []
            idx = self._num_suggestions % self.llambda
            individual = self.population[idx]
            a, b = (self.population[np.random.randint(self.llambda)] for _ in range(2))
            if any(x is None for x in [individual, a, b]):
                new_individual = np.random.normal(0, 1, self.hp_num)
                # if (self.trials.trials_num) <= self.llambda:
                assert self.candidates[idx] is None
                self.candidates[idx] = tuple(new_individual)
                self.population[idx] = new_individual
            else:
                new_individual = individual + self.F1 * (a - b) + self.F2 * (self.current_best - individual)
                for i in range(self.hp_num):
                    R = np.random.randint(self.hp_num)
                    if i != R and np.random.uniform(0, 1) > self.CR:
                        new_individual[i] = individual[i]
                self.candidates[idx] = tuple(new_individual)
                # self.population[idx] = new_individual
            self._num_suggestions += 1
            sas.append(new_individual)
            x_arrays.append(self.feature_to_array(np.asarray(new_individual), self.sparse_dimension))

        x = [DenseConfiguration.array_to_dict(self.space,
                                                  np.array(sa)) for sa in x_arrays]
        self.trials.params_history.extend(x)
        # self._num_suggestions += n_suggestions
        return x, sas

    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1

        for xx, yy in zip(x, y):
            idx = self.candidates.index(tuple(xx))
            self.population[idx] = np.asarray(xx)
            self.population_fitness[idx] = yy
            # self._num_suggestions += 1
            self.candidates[idx] = None
            if yy < self.current_best_fitness:
                self.current_best = self.population[idx]
                self.current_best_fitness = yy


opt_class = DE
