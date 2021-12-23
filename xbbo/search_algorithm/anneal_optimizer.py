import numpy as np
import tqdm, random
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         CategoricalHyperparameter,
                                         OrdinalHyperparameter)
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration
from xbbo.core.stochastic import Category, Uniform
from xbbo.core.trials import Trials


class Anneal(AbstractOptimizer):

    def __init__(self,
                 config_spaces,
                 min_sample=1,
                 avg_best_idx=2.0,
                 shrink_coef=0.1):
        AbstractOptimizer.__init__(self, config_spaces)
        self.min_sample = min_sample
        self.avg_best_idx = avg_best_idx
        self.shrink_coef = shrink_coef
        configs = self.space.get_hyperparameters()
        self.nodes = []
        self.node_trial_num = [] # for anneal
        for config in configs:
            self.node_trial_num.append(0)
            if isinstance(config, CategoricalHyperparameter):
                self.nodes.append(Category(config.num_choices))
            elif isinstance(config, OrdinalHyperparameter):
                self.nodes.append(Category(config.num_elements))
            else:
                self.nodes.append(Uniform(0, 1))
        self.hp_num = len(configs)
        self.trials = Trials()

    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if (self.trials.trials_num) <= self.min_sample :
            return self._random_suggest()
        else:
            sas = []
            for n in range(n_suggestions):
                suggest_array = []
                for i in range(self.hp_num):
                    self.node_trial_num[i] += 1
                    best_val = self._choose_best_trail_feature(i)
                    if isinstance(self.nodes[i], Uniform):
                        suggest_array.append(
                            self.nodes[i].sample(*self._handle_uniform(best_val, i))
                        )
                    elif isinstance(self.nodes[i], Category):
                        suggest_array.append(
                            self.nodes[i].sample(self._handle_category(best_val, i))
                        )

                sas.append(suggest_array)
        x = [DenseConfiguration.array_to_dict(self.space,
                                             np.array(sa)) for sa in sas]
        self.trials.params_history.extend(x)
        return x, sas

    def _shrinking(self, node_idx):
        T = self.node_trial_num[node_idx]
        return 1.0 / (1.0 + T * self.shrink_coef)

    def _handle_category(self, val, node_idx):
        val1 = np.atleast_1d(val)
        counts = np.bincount(val1, minlength=self.nodes[node_idx].choices) / val1.size
        prior = self._shrinking(node_idx)
        return (1-prior) * counts + prior * 1/self.nodes[node_idx].choices

    def _handle_uniform(self, midpt, node_idx, low=0, high=1):
        width = (high - low) * self._shrinking(node_idx)

        half = 0.5 * width
        min_midpt = low + half # 避免超出边界
        max_midpt = high - half
        clipped_midpt = np.clip(midpt, min_midpt, max_midpt)
        return clipped_midpt - half, clipped_midpt + half


    def _random_suggest(self, n_suggestions=1):
        sas = []
        for n in range(n_suggestions):
            suggest_array = [node.random_sample() for node in self.nodes]
            for i in range(self.hp_num):
                self.node_trial_num[i] += 1
            sas.append(suggest_array)
        x = [DenseConfiguration.array_to_dict(self.space,
                                             np.array(sa)) for sa in sas]
        self.trials.params_history.extend(x)
        return x, sas


    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1

    def _choose_best_trail_feature(self, node_idx):
        good_idx = np.random.geometric(1.0/self.avg_best_idx) - 1
        good_idx = np.clip(good_idx, 0, self.trials.trials_num - 1).astype("int32")
        picks = np.argsort(self.trials.history_y)[good_idx]
        return np.array(self.trials.history)[picks][node_idx]

opt_class = Anneal