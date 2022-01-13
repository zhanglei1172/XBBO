import glob
import typing
import numpy as np
import torch
from xbbo.acquisition_function.ei import EI
from xbbo.acquisition_function.transfer.taf import TAF_
from xbbo.configspace.feature_space import FeatureSpace_uniform
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration

from xbbo.core.trials import Trial, Trials
from xbbo.surrogate.transfer.weight_stategy import RGPE_mean_surrogate_
from xbbo.surrogate.transfer.tst import TST_surrogate_


class SMBO(AbstractOptimizer, FeatureSpace_uniform):

    def __init__(self,
                 config_spaces,
                 min_sample=4,
                 noise_std=0.01,
                 rho=0.75,
                 bandwidth=0.9,
                 mc_samples=256,
                 raw_samples=100,
                 purn=True,
                 alpha=0
                 # avg_best_idx=2.0,
                 # meta_data_path=None,
                 ):
        AbstractOptimizer.__init__(self, config_spaces)
        FeatureSpace_uniform.__init__(self, self.space.dtypes_idx_map)
        self.min_sample = min_sample
        self.candidates = None
        # self.avg_best_idx = avg_best_idx
        # self.meta_data_path = meta_data_path
        configs = self.space.get_hyperparameters()
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)

        self.hp_num = len(configs)
        self.bounds_tensor = torch.stack([
            torch.zeros(self.hp_num),
            torch.ones(self.hp_num)
        ])
        self.trials = Trials()
        # self.surrogate = GaussianProcessRegressor(self.hp_num)
        self.surrogate = GaussianProcessRegressorARD_torch(self.hp_num, self.min_sample)
        self.acq_class = TAF_
        self.noise_std = noise_std
        self.rho = rho
        self.bandwidth = bandwidth
        self.mc_samples = mc_samples
        self.raw_samples = raw_samples
        self.purn = purn
        self.alpha = alpha


    def suggest(self, n_suggestions=1):
        trial_list = []
        # currently only suggest one
        if (self.trials.trials_num) < self.init_budget:
            assert self.trials.trials_num % n_suggestions == 0
            configs = self.initial_design_configs[
                int(n_suggestions *
                    self.trials.trials_num):int(n_suggestions *
                                                (self.trials.trials_num + 1))]
            for config in configs:
                trial_list.append(
                    Trial(configuration=config,
                          config_dict=config.get_dictionary(),
                          sparse_array=config.get_sparse_array()))
        else:
            self.surrogate_model.update_weight(self._get_similarity())
            self.surrogate_model.train(
                np.asarray(self.trials.get_sparse_array()),
                np.asarray(self.trials.get_history()[0]))
            configs = []
            _, best_val = self._get_x_best(self.predict_x_best)
            self.acquisition_func.update(surrogate_model=self.surrogate_model,
                                         y_best=best_val)
            configs = self.acq_maximizer.maximize(self.trials,
                                                  1000,
                                                  drop_self_duplicate=True)
            _idx = 0
            for n in range(n_suggestions):
                while _idx < len(configs):  # remove history suggest
                    if not self.trials.is_contain(configs[_idx]):
                        config = configs[_idx]
                        configs.append(config)
                        trial_list.append(
                            Trial(configuration=config,
                                  config_dict=config.get_dictionary(),
                                  sparse_array=config.get_sparse_array()))
                        _idx += 1

                        break
                    _idx += 1
                else:
                    assert False, "no more configs can be suggest"
                # surrogate = TST_surrogate(self.gps, self.target_model,
                #   self.similarity, self.rho)

        return trial_list


    def observe(self, trial_list):
        # print(y)
        for trial in trial_list:
            self.trials.add_a_trial(trial)

    def _get_x_best(self, predict: bool) -> typing.Tuple[float, np.ndarray]:
        """Get value, configuration, and array representation of the "best" configuration.

        The definition of best varies depending on the argument ``predict``. If set to ``True``,
        this function will return the stats of the best configuration as predicted by the model,
        otherwise it will return the stats for the best observed configuration.

        Parameters
        ----------
        predict : bool
            Whether to use the predicted or observed best.

        Returns
        -------
        float
        np.ndarry
        Configuration
        """
        if predict:
            X = self.trials.get_sparse_array()
            costs = list(
                map(
                    lambda x: (
                        self.surrogate_model.predict(x.reshape((1, -1)))[0][0],
                        x,
                    ),
                    X,
                ))
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            best_idx = self.trials.best_id
            x_best_array = self.trials.get_sparse_array()[best_idx]
            best_observation = self.trials.best_observe_value

        return x_best_array, best_observation



opt_class = SMBO
