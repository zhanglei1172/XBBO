import numpy as np
import cma
from bbomark.configspace.feature_space import FeatureSpace_gaussian
from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations
from bbomark.core.trials import Trials


class CMA(AbstractOptimizer, FeatureSpace_gaussian):

    def __init__(self,
                 config_spaces,):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        # configs = self.space.get_hyperparameters()
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.es = cma.CMAEvolutionStrategy([0.] * self.dense_dimension, 1.0)
        # self.hp_num = len(configs)

        self.trials = Trials()
        self.listx = []
        self.listy = []

    def suggest(self, n_suggestions=1):

        sas = []
        x_arrays = []
        for n in range(n_suggestions):
            new_individual = self.es.ask(1)[0]
            sas.append(new_individual)
            x_arrays.append(self.feature_to_array(np.asarray(new_individual), self.sparse_dimension))

        x = [Configurations.array_to_dictUnwarped(self.space,
                                                  np.array(sa)) for sa in x_arrays]
        self.trials.params_history.extend(x)
        # self._num_suggestions += n_suggestions
        return x, sas

    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1
        self.listx += x
        self.listy += y
        if len(self.listx) >= self.es.popsize:
            self.es.tell(self.listx, self.listy)
            self.listx, self.listy = [], []



opt_class = CMA
