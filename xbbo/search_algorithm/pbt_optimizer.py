import copy
import math
from typing import Optional, List, Tuple, cast

import numpy as np

from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace, convert_denseConfigurations_to_array
from xbbo.core.trials import Trials
from xbbo.initial_design import ALL_avaliable_design
from . import alg_register

@alg_register.register('pbt')
class PBT():
    def __init__(self,
                 space: DenseConfigurationSpace,
                 pop_size: int = 30,
                 initial_design: str = 'random',
                 seed: int = 42,
                 fraction: float = 0.2,
                 **kwargs):
        # FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        
        self.space = space
        self.rng = np.random.RandomState(seed)
        self.dimension = self.space.get_dimensions()
        self.pop_size = pop_size if pop_size else 4 + math.floor(3 * math.log(self.dimension))
        self.pop_size = pop_size
        self.init_budget = pop_size

        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, init_budget=self.init_budget)

        # self.population_hp = [{'h1':1.0, 'h2':0.0}, {'h1':0.0,'h2':1.0}]
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        self.population_configs = self.initial_design_configs
        self.population_hp_array = convert_denseConfigurations_to_array(self.population_configs)
        self.trials = Trials(dim=self.dimension)

        # self.data_shuffle_seed = kwargs.get('seed', 0)
        self.fraction = fraction
        self.population_losses_his = []

    def init_model_hp(self, population_model):
        for i, model in enumerate(population_model):
            model.loss = model.evaluate()
            model.update_hp(self.population_configs[i].get_dictionary())
        self.population_losses_his.append(
            [model.loss
             for model in population_model])

    def exploit_and_explore(self, population_model, losses):
        self.population_losses_his.append(losses)
        s_id = np.argsort(losses)
        top_ids = s_id[:int(self.fraction * len(s_id))]
        bot_ids = s_id[-int(self.fraction * len(s_id)):]
        # nobot_ids = s_id[:-self.fraction*len(s_id)]
        for bot_id in bot_ids:
            # exploit
            top_id = self.rng.choice(top_ids)
            checkpoint = population_model[top_id].save_checkpoint()
            population_model[bot_id].load_checkpoint(checkpoint)
            self.population_hp_array[bot_id] = self.population_hp_array[
                top_id].copy()
            # explore
            self.population_hp_array[bot_id] = np.clip(
                self.population_hp_array[bot_id] +
                np.random.normal(0, 0.2, size=self.dimension), 0, 1)

            x_array = self.feature_to_array(self.population_hp_array[bot_id],
                                            self.sparse_dimension)
            x_unwarped = DenseConfiguration.array_to_dict(self.space, x_array)
            self.population_hp[bot_id] = x_unwarped
            population_model[bot_id].history_hp = copy.copy(
                population_model[top_id].history_hp)
            population_model[bot_id].history_loss = copy.copy(
                population_model[top_id].history_loss)
            population_model[bot_id].update_hp(x_unwarped)

    def exploit_and_explore_toy(self, population_model, losses):
        self.population_losses_his.append(losses)
        s_id = np.argsort(losses)
        top_num = max(int(self.fraction * len(s_id)), 1)
        top_ids = s_id[:top_num]
        bot_ids = s_id[-top_num:]
        # nobot_ids = s_id[:-self.fraction*len(s_id)]
        for bot_id in bot_ids:  # TODO Toy every point do explore
            # exploit
            top_id = np.random.choice(top_ids)
            checkpoint = population_model[top_id].save_checkpoint()
            population_model[bot_id].load_checkpoint(checkpoint)
            # self.population_hp_array[bot_id] = self.population_hp_array[top_id].copy() # TODO Toy
            # explore
            self.population_hp_array[bot_id] = np.clip(  # TODO Toy
                self.population_hp_array[top_id] +
                self.rng.normal(0, 0.1, size=self.dimension), 0, 1)

            
            new_config = DenseConfiguration.from_array(self.space, self.population_hp_array[bot_id])
            self.population_configs[bot_id] = new_config
            # self.population_hp[bot_id] = x_unwarped
            # population_model[bot_id].history_hp = copy.copy(
                # population_model[top_id].history_hp)
            population_model[bot_id].history_loss = copy.copy(
                population_model[top_id].history_loss)
            population_model[bot_id].update_hp(new_config.get_dictionary())


opt_class = PBT
