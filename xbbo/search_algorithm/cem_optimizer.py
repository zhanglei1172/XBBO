import numpy as np

from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from . import alg_register
# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.core.trials import Trial, Trials


@alg_register.register('cem')
class CEM(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 llambda=30,
                 elite_ratio=0.3,
                 sample_method: str = 'Gaussian',
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   **kwargs)
        # Uniform2Gaussian.__init__(self, )

        # configs = self.space.get_hyperparameters()
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()
        if sample_method == 'Gaussian':
            self.sampler = Gaussian_sampler(self.dimension, self.bounds,
                                            self.rng)
        elif sample_method == 'Uniform':
            self.sampler = Uniform_sampler(self.dimension, self.bounds,
                                           self.rng)

        self.buffer_x = []
        self.buffer_y = []
        self.llambda = llambda  #if llambda else 4 + math.floor(3 * math.log(self.dimension))
        self.elite_ratio = elite_ratio
        self.elite_num = max(int(round(self.llambda * self.elite_ratio)), 2)
        self.trials = Trials(dim=self.dimension)

    def _suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            # new_individual = self.feature_to_array(new_individual, )
            new_individual = self.sampler.sample()

            config = DenseConfiguration.from_array(self.space, new_individual)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=new_individual,
                      origin='CEM'))

        return trial_list

    def _get_elite(self):
        self.buffer_x = np.asarray(self.buffer_x)
        self.buffer_y = np.asarray(self.buffer_y)
        idx = np.argsort(self.buffer_y)[:self.elite_num]
        return self.buffer_x[idx, :], self.buffer_y[idx]

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            self.buffer_x.append(trial.array)
            self.buffer_y.append(trial.observe_value)
        if len(self.buffer_x) < self.llambda:
            return
        elite_x, elite_y = self._get_elite()
        self.sampler.update(elite_x, elite_y)
        self.buffer_x = []
        self.buffer_y = []


class Gaussian_sampler():
    def __init__(self, dim, bounds, rng) -> None:
        self.bounds = bounds
        u = bounds.ub
        l = bounds.lb
        self.mean = (u + l) / 2
        self.std = (u - l) / 6
        self.dim = dim
        self.rng = rng

    def update(self, elite_x, elite_y):
        self.mean = np.mean(elite_x, axis=0)
        self.std = np.std(elite_x, axis=0)

    def sample(self, ):
        new_individual = self.rng.normal(self.mean, self.std + 1e-17)
        new_individual = np.clip(new_individual, self.bounds.lb,
                                 self.bounds.ub)
        return new_individual


class Uniform_sampler():
    def __init__(self, dim, bounds, rng) -> None:
        self.bounds = bounds
        u = bounds.ub
        l = bounds.lb
        self.min = np.copy(l)
        self.max = np.copy(u)
        self.dim = dim
        self.rng = rng

    def update(self, elite_x, elite_y):
        self.min = np.amin(elite_x, axis=0)
        self.max = np.amax(elite_x, axis=0)

    def sample(self, ):
        new_individual = self.rng.uniform(self.min, self.max)
        new_individual = np.clip(new_individual, self.bounds.lb,
                                 self.bounds.ub)
        return new_individual


opt_class = CEM