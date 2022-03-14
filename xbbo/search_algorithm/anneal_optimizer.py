# from collections import OrderedDict
import numpy as np

from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
# from xbbo.core.stochastic import Category, Uniform
from xbbo.core.trials import Trial, Trials
from . import alg_register
from xbbo.initial_design import ALL_avaliable_design


@alg_register.register('anneal')
class Anneal(AbstractOptimizer):
    def __init__(self,
                 space,
                 seed: int = 42,
                 suggest_limit=10,
                 initial_design: str = 'sobol',
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)
        self.dimension = self.space.get_dimensions()

        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )

        self.avg_best_idx = kwargs.get("avg_best_idx", 2.0)
        self.shrink_coef = kwargs.get("shrink_coef", 0.1)
        self.cat_nodes = []
        self.num_nodes = []
        for k in self.space.map:
            if k != 'num':
                self.cat_nodes.extend([
                    {
                        "idx": self.space.map[k].src[i],
                        # "activate_num": 0,
                        "n_choice": self.space.map[k].sizes[i]
                    } for i in range(len(self.space.map[k].src))
                ])
            else:
                self.num_nodes.extend([{
                    "idx": self.space.map[k].src[i],
                } for i in range(len(self.space.map[k].src))])

        self.trials = Trials(dim=self.dimension)

    def _suggest(self, n_suggestions=1):
        trial_list = []
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
                          array=config.get_array(sparse=False)))
        else:
            for n in range(n_suggestions):
                X = self.trials.get_array()
                Y = np.asarray(self.trials._his_observe_value)

                array = np.empty(self.dimension)
                for node in self.cat_nodes:
                    idx = node["idx"]
                    x_ = X[:, idx].astype(
                        'int')  # deactivate conditional variable
                    mask = ~np.isnan(x_)
                    x = x_[mask]
                    y = Y[mask]
                    best_val = self._choose_best_trail_feature(x, y)
                    X[:, idx]
                    p = self._handle_category(best_val, len(y),
                                              node["n_choice"])
                    array[idx] = self.rng.choice(node["n_choice"], p=p)
                for node in self.num_nodes:
                    idx = node["idx"]
                    x_ = X[:, idx]  # deactivate conditional variable
                    mask = ~np.isnan(x_)
                    x = x_[mask]
                    y = Y[mask]
                    best_val = self._choose_best_trail_feature(x, y)
                    # X[:, idx]
                    low, high = self._handle_uniform(best_val, len(y))
                    array[idx] = self.rng.uniform(low, high)
                config = DenseConfiguration.from_array(self.space, array)
                array = config.get_array(sparse=False)
                trial_list.append(
                    Trial(configuration=config,
                          config_dict=config.get_dictionary(),
                          array=array))
                # valid_indices = np.argwhere(~np.isnan(sparse_array)).ravel()
                # for node in self.cat_nodes:
                #     if node["idx"] in valid_indices:
                #         node["activate_num"] += 1
                # for node in self.cat_nodes:
                #     if node["idx"] in valid_indices:
                #         node["activate_num"] += 1

        return trial_list

    def _shrinking(self, activate_num):
        T = activate_num
        return 1.0 / (1.0 + T * self.shrink_coef)

    def _handle_category(self, val, activate_num, n_choice):
        prior = self._shrinking(activate_num)
        if val is None:
            return np.ones(n_choice) / n_choice
        val1 = np.atleast_1d(val)
        counts = np.bincount(val1, minlength=n_choice) / val1.size
        return (1 - prior) * counts + prior * 1 / n_choice

    def _handle_uniform(self, midpt, activate_num, low=0, high=1):
        if midpt is None:
            return low, high
        width = (high - low) * self._shrinking(activate_num)

        half = 0.5 * width
        min_midpt = low + half  # 避免超出边界
        max_midpt = high - half
        clipped_midpt = np.clip(midpt, min_midpt, max_midpt)
        return clipped_midpt - half, clipped_midpt + half

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)

    def _choose_best_trail_feature(self, his_x, his_y):
        if len(his_y) <= self.avg_best_idx:
            return None
        good_idx = self.rng.geometric(1.0 / self.avg_best_idx) - 1
        good_idx = np.clip(good_idx, 0,
                           self.trials.trials_num - 1).astype("int32")
        picks = np.argsort(his_y)[good_idx]
        return his_x[picks]


opt_class = Anneal