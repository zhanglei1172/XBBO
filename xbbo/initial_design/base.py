from collections import OrderedDict
import typing
import numpy as np
from ConfigSpace.hyperparameters import NumericalHyperparameter, \
    Constant, CategoricalHyperparameter, OrdinalHyperparameter

from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace, deactivate_inactive_hyperparameters


class InitialDesign:
    '''
    reference: https://github.com/automl/SMAC3/blob/master/xbbo/initial_design/initial_design.py
    '''
    def __init__(self,
                 cs: DenseConfigurationSpace,
                 rng: np.random.RandomState,
                 ta_run_limit: typing.Optional[int] = None,
                 n_configs_x_params: typing.Optional[int] = 10,
                 init_budget: typing.Optional[int] = None,
                 max_config_fracs: float = 0.25,
                 **kwargs) -> None:
        self.cs = cs
        self.dim = cs.get_dimensions()
        self.rng = rng
        # n_params = len(self.cs.get_hyperparameters())
        if init_budget is not None:
            self.init_budget = init_budget
        elif n_configs_x_params is not None:
            self.init_budget = int(
                max(
                    1,
                    min(n_configs_x_params * self.dim,
                        (max_config_fracs * ta_run_limit))))

    def select_configurations(self) -> typing.List[DenseConfiguration]:
        self.configs = self._select_configurations()
        self.configs = list(OrderedDict.fromkeys(self.configs))
        return self.configs

    def _select_configurations(self, num=None) -> typing.List[DenseConfiguration]:
        raise NotImplementedError

    def _transform_continuous_designs(
            self, design: np.ndarray, origin: str,
            cs: DenseConfigurationSpace) -> typing.List[DenseConfiguration]:

        # params = cs.get_hyperparameters()
        # for idx, param in enumerate(params):
        #     if isinstance(param, NumericalHyperparameter):
        #         continue
        #     elif isinstance(param, Constant):
        #         # add a vector with zeros
        #         design_ = np.zeros(np.array(design.shape) + np.array((0, 1)))
        #         design_[:, :idx] = design[:, :idx]
        #         design_[:, idx + 1:] = design[:, idx:]
        #         design = design_
        #     elif isinstance(param, CategoricalHyperparameter):
        #         v_design = design[:, idx]
        #         v_design[v_design == 1] = 1 - 10**-10
        #         design[:, idx] = np.array(v_design * len(param.choices),
        #                                   dtype=np.int)
        #     elif isinstance(param, OrdinalHyperparameter):
        #         v_design = design[:, idx]
        #         v_design[v_design == 1] = 1 - 10**-10
        #         design[:, idx] = np.array(v_design * len(param.sequence),
        #                                   dtype=np.int)
        #     else:
        #         raise ValueError("Hyperparameter not supported in LHD")

        configs = []
        for vector in design:
            conf = DenseConfiguration.from_array(cs, vector)
            # conf = deactivate_inactive_hyperparameters(configuration=None,
            #                                            configuration_space=cs,
            #                                            vector=vector)
            conf.origin = origin
            configs.append(conf)

        return configs
