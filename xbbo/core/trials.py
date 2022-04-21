from typing import Iterable
import numpy as np
from xbbo.configspace.space import DenseConfiguration
from xbbo.utils.constants import Key


class Trial:
    def __init__(self,
                 configuration,
                 config_dict,
                 observe_value=None,
                 array=None,
                #  sparse_array=None,
                 time=None,
                 origin: str = '',
                 marker=None,
                 info: dict=None,
                 **kwargs) -> None:
        self.config_dict = config_dict
        self.array = array
        # self.sparse_array = sparse_array
        self.configuration = configuration
        self.observe_value = observe_value
        self.time = time
        self.origin = origin
        self.marker = marker
        self.info = info if info else {}
        for k in kwargs:
            setattr(self, k, kwargs[k])

    def add_observe_value(self, observe_value=None, obs_info=None):
        if obs_info is None:
            obs_info = {}
        self.info.update(obs_info)
        self.observe_value = observe_value


class Trials:
    def __init__(self, dim):
        self._his_hash_configs_set = set()
        self._his_configs_set = set()
        self._his_configs = []
        self._his_array = np.empty((0, dim))
        self._his_observe_value = []
        self._his_configs_dict = []
        self.best_observe_value = np.inf
        self.best_id = None
        self.trials_num = 0
        self.markers = []
        self.infos = []
        # self.run_id = 0
        # self.run_history = {}
        self.traj_history = []
        # self.use_dense = use_dense

    def add_a_trial(self, trial: Trial, permit_duplicate=True):
        hash_config = str(trial.configuration)+str(trial.info.get(Key.BUDGET, 'max_budget'))
        if not permit_duplicate:
            assert hash_config not in self._his_hash_configs_set
        self.infos.append(trial.info)
        self._his_hash_configs_set.add(hash_config)
        self._his_configs_set.add(trial.configuration)
        self._his_configs.append(trial.configuration)
        self.traj_history.append(trial)
        self._his_configs_dict.append(trial.config_dict)
        self._his_observe_value.append(trial.observe_value)
        self.markers.append(trial.marker)
        # if self.use_dense:
        #     assert trial.dense_array is not None
        # else:
        #     assert trial.sparse_array is not None
        if trial.array is not None:
            self._his_array = np.vstack(
                [self._his_array, trial.array])
        # if trial.sparse_array is not None:
        #     self._his_sparse_array = np.vstack(
        #         [self._his_sparse_array, trial.sparse_array])
        obs = sum(trial.observe_value) if isinstance(trial.observe_value, Iterable) else trial.observe_value
        if self.best_observe_value > obs:
            self.best_observe_value = obs
            self.best_id = self.trials_num
        self.trials_num += 1

    def get_array(self):
        if len(self._his_array) == self.trials_num:
            return self._his_array
        self._his_array = [
            config.get_array(sparse=False) for config in self._his_configs
        ]
        return self._his_array

    # def get_sparse_array(self):
    #     if len(self._his_sparse_array) == self.trials_num:
    #         return self._his_sparse_array
    #     self._his_sparse_array = [
    #         config.get_sparse_array() for config in self._his_configs
    #     ]
    #     return self._his_sparse_array

    def add_trials(self, trials):
        for trial in trials._traj_history:
            self.add_a_trial(trial)

    def is_contain(self, config: DenseConfiguration) -> bool:
        return config in self._his_hash_configs_set

    def is_empty(self, ):
        return self.trials_num == 0

    def get_all_configs(self, ):
        return self._his_configs

    def get_best(self):
        return self.best_observe_value, self._his_configs_dict[self.best_id]

    def get_history(self):
        return self._his_observe_value, self._his_configs_dict
    
    # def get_array(self):
    #     if self.use_dense:
    #         return self.get_dense_array()
    #     else:
    #         return self.get_sparse_array()

    # def visualize(self, ax=None):
    #     if ax is None:
    #         _, ax = plt.subplots(111)
    #     ax.plot(np.minimum.acumalate(self._his_observe_value))