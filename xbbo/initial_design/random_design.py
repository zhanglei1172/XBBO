# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

from typing import List

from xbbo.initial_design.base import InitialDesign
from xbbo.configspace.space import DenseConfiguration

class RandomDesign(InitialDesign):
    """Initial design that evaluates random configurations."""

    def _select_configurations(self, num=None) -> List[DenseConfiguration]:
        """Select a random configuration.

        Returns
        -------
        config: Configuration()
            Initial incumbent configuration
        """
        design_num = num if num else self.init_budget
        configs = self.cs.sample_configuration(size=design_num)
        if design_num == 1:
            configs = [configs]
        for config in configs:
            config.origin = 'Random initial design.'
        return configs
