import typing
import numpy as np
from ConfigSpace.hyperparameters import NumericalHyperparameter, \
    Constant, CategoricalHyperparameter, OrdinalHyperparameter
from ConfigSpace.util import deactivate_inactive_hyperparameters

from bbomark.configspace.space import Configurations, Space


class InitialDesign:
    '''
    reference: https://github.com/automl/SMAC3/blob/master/smac/initial_design/initial_design.py
    '''
    def __init__(
        self,
        dim: int,
        rng,
        ta_run_limit: int,
        n_configs_x_params: typing.Optional[int] = 10,
        init_budget: typing.Optional[int] = None,
        max_config_fracs: float = 0.25,
    ) -> None:
        self.dim = dim
        self.rng = rng
        # n_params = len(self.cs.get_hyperparameters())
        if init_budget is not None:
            self.init_budge = init_budget
        elif n_configs_x_params is not None:
            self.init_budget = int(
                max(
                    1,
                    min(n_configs_x_params * self.dim,
                        (max_config_fracs * ta_run_limit))))

    def select_configurations(self) -> typing.List[Configurations]:
        self.configs = self._select_configurations()
        return self.configs

    def _select_configurations(self) -> typing.List[Configurations]:
        raise NotImplementedError


