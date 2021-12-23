import numpy as np

from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration
from xbbo.configspace.feature_space import FeatureSpace_gaussian
from xbbo.core import AbstractOptimizer

class CEM(AbstractOptimizer, FeatureSpace_gaussian):
    def __init__(self, config_spaces):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)

        # configs = self.space.get_hyperparameters()
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)

        self.hp_range = []
        for k in range(self.dense_dimension):
            self.hp_range.append([0, 1])
        self.mean = np.zeros(self.dense_dimension)+0.5
        self.std = np.ones(self.dense_dimension)
        self.buffer_x = []
        self.buffer_y = []
        self.lam = 10
        self.elite_ratio = 0.3
        self.history_y = []

    def suggest(self, n_suggestions=1):
        sas = []
        x = []
        for n in range(n_suggestions):
            suggest_array = np.random.normal(self.mean, self.std)
            for d in range(self.dense_dimension):
                suggest_array[d] = suggest_array[d].clip(self.hp_range[d][0],
                                                         self.hp_range[d][1])
            sas.append((suggest_array))
            x_array = self.feature_to_array(suggest_array, self.sparse_dimension)

            x.append(DenseConfiguration.array_to_dict(self.space, x_array))

        return x, sas

    def _get_elite(self):
        self.buffer_x = np.asarray(self.buffer_x)
        self.buffer_y = np.asarray(self.buffer_y)
        idx = np.argsort(self.buffer_y)[:int(self.lam * self.elite_ratio)]
        return self.buffer_x[idx, :], self.buffer_y[idx]

    def observe(self, x, y):
        for n_suggest in range(len(x)):
            suggest_array = (x[n_suggest])
            self.buffer_x.append(suggest_array)
            self.buffer_y.append(y[n_suggest])
            self.history_y.append(y[n_suggest])
        if len(self.buffer_x) < self.lam:
            return
        elite_x, elite_y = self._get_elite()
        self.mean = np.mean(elite_x, axis=0)
        self.std = np.std(elite_x, axis=0)
        self.buffer_x = []
        self.buffer_y = []




opt_class = CEM