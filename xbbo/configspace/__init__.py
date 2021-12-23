# from copy import deepcopy

from .space import DenseConfigurationSpace
import ConfigSpace.hyperparameters as CSH

# from .warp import Warp


def build_space(meta, seed=42):
    """Build Real configspace class.

    Parameters
    ----------
    meta : dict(str, dict)
        Configuration of variables in joint configspace. See API description.
    """
    cs = DenseConfigurationSpace(seed=seed, meta=meta)
    # warp = Warp()
    param_list = sorted(meta.keys())
    # all_warp = {}
    for param_name in param_list:
        # param_name = deepcopy(param_name_)
        param_config = meta[param_name]

        param_type = param_config["type"]

        param_space = param_config.get("warp", 'linear')
        param_range = param_config.get("range", None)
        param_values = param_config.get("values", None)

        # prewarp = 'linear' # None
        if param_type == "cat":  # one-hot
            assert param_range is None
            cs.add_hyperparameter(
                CSH.CategoricalHyperparameter(name=param_name,
                                              choices=param_values))
        elif param_type == "bool":
            assert param_range is None
            assert param_values is None

            cs.add_hyperparameter(
                CSH.CategoricalHyperparameter(name=param_name,
                                              choices=[False, True]))
        elif param_values is not None:
            assert param_type in ("int", "ord", "float"
                                  )  # TODO 有些优化器能直接处理Ordinal？现在直接转为real，warp取整
            assert param_space == 'linear'
            cs.add_hyperparameter(
                CSH.CategoricalHyperparameter(name=param_name,
                                              choices=param_values))
        elif param_type == 'int':
            assert param_values is None

            cs.add_hyperparameter(
                CSH.UniformIntegerHyperparameter(name=param_name,
                                                 lower=param_range[0],
                                                 upper=param_range[1],
                                                 log=param_space == 'log'))
        elif param_type == "float":
            assert param_values is None
            assert param_range is not None
            cs.add_hyperparameter(
                CSH.UniformFloatHyperparameter(name=param_name,
                                               lower=param_range[0],
                                               upper=param_range[1],
                                               log=param_space == 'log'))
        else:
            assert False, "type %s not handled in API" % param_type

    # config_spaces =
    return cs