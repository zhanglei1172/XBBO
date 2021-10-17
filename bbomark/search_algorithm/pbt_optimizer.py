import copy
import functools
import math
import random
from typing import Optional, List, Tuple, cast

import numpy as np

from bbomark.configspace.feature_space import FeatureSpace_gaussian, FeatureSpace_uniform
from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations
from bbomark.core.trials import Trials


class PBT(FeatureSpace_uniform):

    def __init__(self,
                 config_spaces, pop_size, **kwargs):
        self.space = config_spaces
        # FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        FeatureSpace_uniform.__init__(self, self.space.dtypes_idx_map)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.pop_size = pop_size
        # TODO Toy
        self.population_hp = [self.space.sample_configuration()[0].get_dict_unwarped() for _ in range(pop_size)]
        # self.population_hp = [{'h1':1.0, 'h2':0.0}, {'h1':0.0,'h2':1.0}]
        self.population_hp_array = [
            self.array_to_feature(Configurations.dictUnwarped_to_array(self.space, hp), self.dense_dimension) for hp
            in self.population_hp]

        self.data_shuffle_seed = kwargs.get('seed', 0)
        self.fraction = kwargs.get("fraction", 0.2)
        self.population_scores_his = []

    def init_model_hp(self, population_model):
        for i, model in enumerate(population_model):
            model.acc = model.evaluate()
            model.update_hp(self.population_hp[i])
        self.population_scores_his.append([model.acc for model in population_model])

    def exploit_and_explore(self, population_model, scores):
        self.population_scores_his.append(scores)
        s_id = np.argsort(scores)[::-1]
        top_ids = s_id[:int(self.fraction * len(s_id))]
        bot_ids = s_id[-int(self.fraction * len(s_id)):]
        # nobot_ids = s_id[:-self.fraction*len(s_id)]
        for bot_id in bot_ids:
            # exploit
            top_id = np.random.choice(top_ids)
            checkpoint = population_model[top_id].save_checkpoint()
            population_model[bot_id].load_checkpoint(checkpoint)
            self.population_hp_array[bot_id] = self.population_hp_array[top_id].copy()
            # explore
            self.population_hp_array[bot_id] = np.clip(
                self.population_hp_array[bot_id] + np.random.normal(0, 0.2, size=self.dense_dimension), 0, 1)

            x_array = self.feature_to_array(self.population_hp_array[bot_id], self.sparse_dimension)
            x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            self.population_hp[bot_id] = x_unwarped
            population_model[bot_id].history_hp = copy.copy(population_model[top_id].history_hp)
            population_model[bot_id].history_score = copy.copy(population_model[top_id].history_score)
            population_model[bot_id].update_hp(x_unwarped)

    def exploit_and_explore_toy(self, population_model, scores):
        self.population_scores_his.append(scores)
        s_id = np.argsort(scores)[::-1]
        top_ids = s_id[:int(self.fraction * len(s_id))]
        bot_ids = s_id[-int(self.fraction * len(s_id)):]
        # nobot_ids = s_id[:-self.fraction*len(s_id)]
        for bot_id in bot_ids: # TODO Toy every point do explore
            # exploit
            top_id = np.random.choice(top_ids)
            checkpoint = population_model[top_id].save_checkpoint()
            population_model[bot_id].load_checkpoint(checkpoint)
            # self.population_hp_array[bot_id] = self.population_hp_array[top_id].copy() # TODO Toy
            # explore
            self.population_hp_array[bot_id] = np.clip( # TODO Toy
                self.population_hp_array[top_id] + np.random.normal(0, 0.1, size=self.dense_dimension), 0, 1)

            x_array = self.feature_to_array(self.population_hp_array[bot_id], self.sparse_dimension)
            x_unwarped = Configurations.array_to_dictUnwarped(self.space, x_array)
            self.population_hp[bot_id] = x_unwarped
            population_model[bot_id].history_hp = copy.copy(population_model[top_id].history_hp)
            population_model[bot_id].history_score = copy.copy(population_model[top_id].history_score)
            population_model[bot_id].update_hp(x_unwarped)


opt_class = PBT
