# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

import typing
from scipy.stats.qmc import Sobol
import numpy as np

from xbbo.configspace.space import DenseConfiguration
from xbbo.initial_design.base import InitialDesign
from xbbo.utils.constants import MAXINT


class SobolDesign(InitialDesign):
    """ Sobol sequence design with a scrambled Sobol sequence.

    See https://scipy.github.io/devdocs/reference/generated/scipy.stats.qmc.Sobol.html for further information

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """
    def __init__(self, cs, rng: np.random.RandomState, ta_run_limit: typing.Optional[int] = None, n_configs_x_params: typing.Optional[int] = 10, init_budget: typing.Optional[int] = None, max_config_fracs: float = 0.25, **kwargs) -> None:
        super().__init__(cs, rng, ta_run_limit, n_configs_x_params, init_budget, max_config_fracs, **kwargs)
        if self.init_budget:
            self.init_budget = 2**int(np.log2(self.init_budget))

    def _select_configurations(self, num=None) -> typing.List[DenseConfiguration]:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """
        design_num = num if num else self.init_budget
        sobol_gen = Sobol(d=self.dim,
                          scramble=True,
                          seed=self.rng.randint(low=0, high=MAXINT))
        sobol = sobol_gen.random(design_num)
        return self._transform_continuous_designs(design=sobol,
                                                  origin='Sobol',
                                                  cs=self.cs)