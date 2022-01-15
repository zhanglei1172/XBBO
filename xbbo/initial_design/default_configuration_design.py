# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

from typing import List

from xbbo.configspace.space import DenseConfiguration
from xbbo.initial_design.base import InitialDesign


class DefaultConfiguration(InitialDesign):

    """Initial design that evaluates default configuration"""

    def _select_configurations(self, num=None) -> List[DenseConfiguration]:
        """Selects the default configuration.

        Returns
        -------
        config: Configuration
            Initial incumbent configuration.
        """

        config = self.cs.get_default_configuration()
        config.origin = 'Default'
        self.init_budget = 1
        return [config]
