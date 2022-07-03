from abc import abstractmethod
import copy
import math
from typing import Optional, List, Tuple, cast
from ConfigSpace import ConfigurationSpace
import numpy as np

from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace, convert_denseConfigurations_to_array
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.core.trials import Trials
from xbbo.initial_design import ALL_avaliable_design
from xbbo.utils.util import create_rng
# from . import alg_register

class Abstract_PBT_Model():
    def __init__(self, seed, **kwargs):
        self.rng = create_rng(seed)
        self.step_num = 0
        

    @abstractmethod
    def __len__(self):
        '''
        one epoch contains how many batchs
        '''
        return 1

    @abstractmethod
    def update_hp(self, params: dict):
        pass

    @abstractmethod
    def _one_step(self):
        pass
    
    def step(self, num, **kwargs):  # train need training(optimizer)
        for it in range(num):
            self.step_num += 1
            self._one_step(**kwargs)

    @abstractmethod
    def evaluate(self) -> float:
        '''
        return loss (which should be minimized)
        '''
        pass

    @abstractmethod
    def load_checkpoint(self, checkpoint):
        pass
    
    @abstractmethod
    def save_checkpoint(self):
        return None

    @classmethod
    @abstractmethod
    def get_configuration_space(cls):
        return None



# @alg_register.register('pbt')
class PBT(AbstractOptimizer):
    def __init__(
            self,
            space: ConfigurationSpace,
            init_budget: int = 30,
            initial_design: str = 'random',
            seed: int = 42,
            fraction: float = 0.2,
            # init_budget: int = None,
            #  min_sample=1,
            suggest_limit: int = np.inf,
            **kwargs):
        # FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)

        self.rng = np.random.RandomState(seed)
        self.dimension = self.space.get_dimensions()
        self.init_budget = init_budget if init_budget else 4 + math.floor(
            3 * math.log(self.dimension))
        self.init_budget = init_budget

        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, init_budget=self.init_budget)
        if self.init_budget is None:
            self.init_budget = self.initial_design.init_budget
        # self.population_hp = [{'h1':1.0, 'h2':0.0}, {'h1':0.0,'h2':1.0}]
        self.initial_design_configs = self.initial_design.select_configurations(
        )[:self.init_budget]
        self.population_configs = self.initial_design_configs
        self.population_hp_array = convert_denseConfigurations_to_array(
            self.population_configs)
        self.trials = Trials(space, dim=self.dimension)

        # self.data_shuffle_seed = kwargs.get('seed', 0)
        self.fraction = fraction
        self.population_losses_his = []

    def init_model_hp(self, population_model:List[Abstract_PBT_Model]):
        for i, model in enumerate(population_model):
            model.loss = model.evaluate()
            model.update_hp(self.population_configs[i].get_dictionary())
        self.population_losses_his.append(
            [model.loss for model in population_model])

    @abstractmethod
    def exploit_and_explore(self, population_model, losses):
        pass

    def _suggest(self, n_suggestions):
        raise NotImplementedError
    
    def _observe(self, trial_list: Trials):
        raise NotImplementedError
    
    def optimize(self, population_model:List[Abstract_PBT_Model], epoch_num, interval):
        finished = False
        for i in range(int(epoch_num*len(population_model[0]))):
            while not finished:
                # parallel training with respective config
                for i in range(self.init_budget):
                    population_model[i].step(
                        int(interval * len(population_model[i])))
                    if population_model[i].step_num == int(
                            len(population_model[i]) * epoch_num):
                        finished = True
                # parallel evalueate with respective config
                for i in range(self.init_budget):
                    population_model[i].evaluate()
                losses = [net.loss for net in population_model]
                assert np.any(np.isfinite(losses)), "ERROR: At Least 1 loss is finite"
                if finished:
                    break
                # Update respective config
                self.exploit_and_explore(population_model, losses)
        return losses


opt_class = PBT
