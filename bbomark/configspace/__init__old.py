# from copy import deepcopy

from .space import Space
from ConfigSpace import ConfigurationSpace
import ConfigSpace.hyperparameters as CSH
from .warp import Warp

def build_space(meta):
    """Build Real configspace class.

    Parameters
    ----------
    meta : dict(str, dict)
        Configuration of variables in joint configspace. See API description.
    """
    cs = ConfigurationSpace(seed=1234, meta=meta)
    warp = Warp()
    param_list = sorted(meta.keys())
    # all_warp = {}
    for param_name in param_list:
        # param_name = deepcopy(param_name_)
        param_config = meta[param_name]

        param_type = param_config["type"]

        param_space = param_config.get("configspace", 'linear')
        param_range = param_config.get("range", None)
        param_values = param_config.get("values", None)

        # prewarp = 'linear' # None
        if param_type == "cat": # one-hot
            assert param_range is None
            arg = warp.warp_space('cat',
                   param_name,
                   param_range=None,
                   param_values=param_values,
                   warp=param_space,
                   discrete_method='linear')
        elif param_type == "bool": # 可以Float加round，或者使用Int
            assert param_range is None
            assert param_values is None
            arg = warp.warp_space('ord',
                   param_name,
                   param_range=None,
                   param_values=[False, True],
                   warp=param_space,
                   discrete_method='linear')
            # arg = CSH.OrdinalHyperparameter(name=param_name, sequence=param_values)
        elif param_values is not None:
            assert param_type in ("int", "ordinal", "real") # TODO 有些优化器能直接处理Ordinal？现在直接转为real，warp取整
            assert param_space == 'linear'
            arg = warp.warp_space('ord',
                   param_name,
                   param_range=None,
                   param_values=param_values,
                   warp=param_space,
                   discrete_method='linear')
        elif param_type == "int": # Be careful when (int and log configspace)
            assert param_values is None
            # Need +1 since API in inclusive
            # choices = range(int(param_range[0]), int(param_range[-1]) + 1)
            if param_space == 'linear':
                arg = warp.warp_space('int',
                       param_name,
                       param_range=param_range,
                       param_values=None,
                       warp='linear',
                       discrete_method='linear')
            else:
                arg = warp.warp_space('float',
                       param_name,
                       param_range=param_range,
                       param_values=None,
                       warp=param_space,
                       discrete_method='round')
            # We are throwing away information here, but OrderedDiscrete appears to be invariant to monotonic transformation anyway.
        elif param_type == "real":
            assert param_values is None
            assert param_range is not None
            arg = warp.warp_space('float',
                   param_name,
                   param_range=param_range,
                   param_values=None,
                   warp=param_space,
                   discrete_method='linear')
        else:
            assert False, "type %s not handled in API" % param_type
        cs.add_hyperparameter(arg)
        # all_args[param_name] = arg
        # all_prewarp[param_name] = prewarp


    # config_spaces =
    return Space(cs, warp)