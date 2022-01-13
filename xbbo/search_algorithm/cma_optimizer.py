from typing import Optional, List, Tuple, cast

import numpy as np
import cma
from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trial, Trials
from . import alg_register


@alg_register.register('cma-es')
class CMAES(AbstractOptimizer, Uniform2Gaussian):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 **kwargs):
        AbstractOptimizer.__init__(self, space, seed, **kwargs)
        Uniform2Gaussian.__init__(self, )

        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.es = cma.CMAEvolutionStrategy([0.] * self.dense_dimension, 1.0)
        # self.hp_num = len(configs)

        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)
        self.listx = []
        self.listy = []

    def suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            new_individual = self.es.ask(1)[0]
            dense_array = self.feature_to_array(new_individual)
            config = DenseConfiguration.from_dense_array(self.space,dense_array)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      dense_array=dense_array,
                      origin='cma-es'))

        return trial_list

    def observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)
            self.listx.append(self.array_to_feature(trial.dense_array))
            self.listy.append(trial.observe_value)
        if len(self.listx) >= self.es.popsize:
            self.es.tell(self.listx, self.listy)
            self.listx = []
            self.listy = []



opt_class = CMAES
