import numpy as np
import matplotlib.pyplot as plt

from xbbo.configspace.space import DenseConfiguration


class Trial:
    def __init__(self,
                 observe_value=None,
                 config_dict={},
                 dense_array=None,
                 sparse_array=None,
                 configuration=None,
                 time=None) -> None:
        self.config_dict = config_dict
        self.dense_array = dense_array
        self.sparse_array = sparse_array
        self.configuation = configuration
        self.observe_value = observe_value
        self.time = time

    
    def add_observe_value(self,observe_value, time=None):
        self.time = time
        self.observe_value = observe_value

class Trials:
    def __init__(self, sparse_dim, dense_dim):
        self._his_configs_set = set()
        self._his_configs = []
        self._his_dense_array = np.empty((0,dense_dim))
        self._his_sparse_array = np.empty((0,sparse_dim))
        self._his_observe_value = []
        self._his_configs_dict = []
        self.best_observe_value = np.inf
        self.best_id = None
        self.trials_num = 0
        # self.run_id = 0
        # self.run_history = {}
        self._traj_history = []
    def add_a_trial(self, trial:Trial):
        assert trial.configuation not in self._his_configs_set
        self._his_configs_set.add(trial.configuation)
        self._his_configs.append(trial.configuation)
        self._traj_history.append(trial)
        self._his_configs_dict.append(trial.config_dict)
        self._his_observe_value.append(trial.observe_value)
        if trial.dense_array is not None:
            self._his_dense_array = np.vstack([self._his_dense_array, trial.dense_array])
        if trial.sparse_array is not None:
            self._his_sparse_array = np.vstack([self._his_sparse_array, trial.sparse_array])
        if self.best_observe_value > trial.observe_value:
            self.best_observe_value = trial.observe_value
            self.best_id = self.trials_num
        self.trials_num += 1
    def get_dense_array(self):
        if len(self._his_dense_array) == self.trials_num:
            return self._his_dense_array
        self._his_dense_array = [config.get_dense_array() for config in self._his_configs]

    def get_sparse_array(self):
        if len(self._his_sparse_array) == self.trials_num:
            return self._his_sparse_array
        self._his_sparse_array = [config.get_sparse_array() for config in self._his_configs]
    
    def add_trials(self, trials):
        for trial in trials._traj_history:
            self.add_a_trial(trial)
    def is_contain(self, config:DenseConfiguration)->bool:
        return config in self._his_configs_set
    
    def is_empty(self,):
        return self.trials_num == 0

    def get_all_configs(self,):
        return self._his_configs

    def get_best(self):
        return self.best_observe_value, self._his_configs_dict[self.best_id]

    def get_history(self):
        return self._his_observe_value, self._his_configs_dict
    
    # def visualize(self, ax=None):
    #     if ax is None:
    #         _, ax = plt.subplots(111)
    #     ax.plot(np.minimum.acumalate(self._his_observe_value))