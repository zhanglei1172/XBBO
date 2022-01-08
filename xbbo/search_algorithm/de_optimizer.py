import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial


class DE(AbstractOptimizer):

    def __init__(self,
                 space:DenseConfigurationSpace,
                 seed:int = 42,
                 llambda=10, **kwargs):
        AbstractOptimizer.__init__(self, space, seed, **kwargs)

        # Uniform2Gaussian.__init__(self,)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.bounds = self.space.get_bounds()

        # self.dense_dimension = len(configs)
        self.llambda = llambda
        self.population = [None] * self.llambda
        self.population_fitness = [None] * self.llambda
        # self.candidates = [None] * self.llambda
        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)
        self.current_best = None
        self.current_best_fitness = np.inf
        self._num_suggestions = 0
        self.F1 = kwargs.get('F1', 0.8)
        self.F2 = kwargs.get('F2',0.8)
        self.CR = kwargs.get('CR',0.5)

    def suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            # suggest_array = []
            idx = self._num_suggestions % self.llambda
            individual = self.population[idx]
            a, b = (self.population[self.rng.randint(self.llambda)] for _ in range(2))
            if any(x is None for x in [individual, a, b]): # population not enough
                # new_individual = self.rng.normal(0, 1, self.dense_dimension)
                new_individual = self.rng.normal(self.bounds.lb, self.bounds.ub, self.dense_dimension)
                # if (self.trials.trials_num) <= self.llambda:
                # assert self.candidates[idx] is None
                # self.candidates[idx] = tuple(new_individual)
                self.population[idx] = new_individual
            else:
                new_individual = individual + self.F1 * (a - b) + self.F2 * (self.current_best - individual)
                R = self.rng.randint(self.dense_dimension)
                for i in range(self.dense_dimension):
                    if i != R and self.rng.uniform(0, 1) > self.CR:
                        new_individual[i] = individual[i]
                # self.candidates[idx] = tuple(new_individual)
                # self.population[idx] = new_individual
            self._num_suggestions += 1
            new_individual = np.clip(new_individual, self.bounds.lb,
                                 self.bounds.ub)
            # dense_array = self.feature_to_array(new_individual)
            config = DenseConfiguration.from_dense_array(self.space,new_individual)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      dense_array=new_individual,
                      origin='DE', loc=idx))

        return trial_list

    def observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicagte=True)
            # idx = self.candidates.index(tuple(trial.dense_array))
            idx = trial.loc
            self.population[idx] = trial.dense_array
            self.population_fitness[idx] = trial.observe_value
            # self._num_suggestions += 1
            # self.candidates[idx] = None
            if trial.observe_value < self.current_best_fitness:
                self.current_best = self.population[idx]
                self.current_best_fitness = trial.observe_value


opt_class = DE
