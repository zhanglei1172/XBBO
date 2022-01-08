import numpy as np

from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.core import AbstractOptimizer
from xbbo.core.trials import Trial, Trials


class CEM(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 llambda=10,
                 elite_ratio=0.3,
                 sample_method: str = 'Gaussian',
                 **kwargs):
        AbstractOptimizer.__init__(self, space, seed, **kwargs)
        # Uniform2Gaussian.__init__(self, )

        # configs = self.space.get_hyperparameters()
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.bounds = self.space.get_bounds()
        if sample_method == 'Gaussian':
            self.sampler = Gaussian_sampler(self.dense_dimension, self.bounds,
                                            self.rng)
        elif sample_method == 'Uniform':
            self.sampler = Uniform_sampler(self.dense_dimension, self.bounds,
                                           self.rng)

        self.buffer_x = []
        self.buffer_y = []
        self.llambda = llambda
        self.elite_ratio = elite_ratio
        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)

    def suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            # new_individual = self.feature_to_array(new_individual, )
            new_individual = self.sampler.sample()

            config = DenseConfiguration.from_dense_array(
                self.space, new_individual)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      dense_array=new_individual,
                      origin='CEM'))

        return trial_list

    def _get_elite(self):
        self.buffer_x = np.asarray(self.buffer_x)
        self.buffer_y = np.asarray(self.buffer_y)
        idx = np.argsort(self.buffer_y)[:int(self.llambda * self.elite_ratio)]
        return self.buffer_x[idx, :], self.buffer_y[idx]

    def observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)
            self.buffer_x.append(trial.dense_array)
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