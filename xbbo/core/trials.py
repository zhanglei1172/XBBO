import numpy as np
from numpy.lib.twodim_base import tri

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
        self.configs = set()
        self.his_dense_dense = np.empty((0,dense_dim))
        self.his_sparse_array = np.empty((0,sparse_dim))
        self.his_observe_value = []
        self.his_config_dict = []
        self.best_observe_value = np.inf
        self.best_id = None
        self.trials_num = 0
        # self.run_id = 0
        # self.run_history = {}
        self.traj_history = []
    def add_a_trial(self, trial:Trial):
        assert trial.configuation not in self.configs
        self.configs.add(trial.configuation)
        self.trials_num += 1
        self.traj_history.append(trial)
        self.his_config_dict.append(trial.config_dict)
        self.his_observe_value.append(trial.observe_value)
        if trial.dense_array is not None:
            self.his_dense_array = np.vstack([self.his_dense_array, trial.dense_array])
        if trial.sparse_array is not None:
            self.his_sparse_array = np.vstack([self.his_sparse_array, trial.sparse_array])
        if self.best_observe_value > trial.observe_value:
            self.best_observe_value = trial.observe_value
            self.best_id = len(self.traj_history)
    
    def add_trials(self, trials):
        for trial in trials.traj_history:
            self.add_a_trial(trial)
    def is_contain(self, config:DenseConfiguration)->bool:
        return config in self.configs
    
    def is_empty(self,):
        return self.trials_num == 0

    def get_all_configs(self,):
        return list(self.configs)