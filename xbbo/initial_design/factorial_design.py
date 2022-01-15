# License: 3-clause BSD
# Copyright (c) 2016-2018, Ml4AAD Group (http://www.ml4aad.org/)

import itertools

from ConfigSpace.hyperparameters import Constant, NumericalHyperparameter, \
    CategoricalHyperparameter, OrdinalHyperparameter
import numpy as np


from xbbo.configspace.space import DenseConfiguration, deactivate_inactive_hyperparameters
from xbbo.initial_design.base import InitialDesign


class FactorialInitialDesign(InitialDesign):
    """Factorial initial design

    Attributes
    ----------
    configs : typing.List[Configuration]
        List of configurations to be evaluated
        Don't pass configs to the constructor;
        otherwise factorial design is overwritten
    """
    def _select_configurations(self, num=None) -> DenseConfiguration:
        """Selects a single configuration to run

        Returns
        -------
        config: Configuration
            initial incumbent configuration
        """
        params = self.cs.get_hyperparameters()

        values = []
        mid = []
        for param in params:
            if isinstance(param, Constant):
                v = [param.value]
                mid.append(param.value)
            elif isinstance(param, NumericalHyperparameter):
                v = [param.lower, param.upper]
                mid.append(np.average([param.lower, param.upper]))
            elif isinstance(param, CategoricalHyperparameter):
                v = list(param.choices)
                mid.append(param.choices[0])
            elif isinstance(param, OrdinalHyperparameter):
                v = [param.sequence[0], param.sequence[-1]]
                length = len(param.sequence)
                mid.append(param.sequence[int(length / 2)])
            values.append(v)

        factorial_design = itertools.product(*values)

        self.logger.debug("Initial Design")
        configs = [self.cs.get_default_configuration()]
        # add middle point in space
        conf_dict = dict([(p.name, v) for p, v in zip(params, mid)])
        middle_conf = deactivate_inactive_hyperparameters(conf_dict, self.cs)
        configs.append(middle_conf)

        # add corner points
        for design in factorial_design:
            conf_dict = dict([(p.name, v) for p, v in zip(params, design)])
            conf = deactivate_inactive_hyperparameters(conf_dict, self.cs)
            conf.origin = "Factorial Design"
            configs.append(conf)
            self.logger.debug(conf)
        size = len(configs)
        self.logger.debug("Size of factorial design: %d" % (size))
        self.init_budget = size
        return configs
