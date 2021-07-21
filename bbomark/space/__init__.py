# from copy import deepcopy

from .space import Space
import ConfigSpace.hyperparameters as CSH

def build_space(meta):
    """Build Real space class.

    Parameters
    ----------
    meta : dict(str, dict)
        Configuration of variables in joint space. See API description.
    """
    cs = Space(seed=1234, meta_param=meta)
    param_list = sorted(meta.keys())
    for param_name in param_list:
        # param_name = deepcopy(param_name_)
        param_config = meta[param_name]

        param_type = param_config["type"]

        param_space = param_config.get("space", None)
        param_range = param_config.get("range", None)
        param_values = param_config.get("values", None)

        prewarp = 'linear' # None
        if param_type == "cat":
            assert param_space is None
            assert param_range is None
            arg = CSH.CategoricalHyperparameter(name=param_name, choices=param_values)
        elif param_type == "bool":
            assert param_space is None
            assert param_range is None
            assert param_values is None
            arg = CSH.OrdinalHyperparameter(name=param_name, sequence=[False, True])
        elif param_values is not None:
            assert param_type in ("int", "ordinal", "real")
            arg = CSH.OrdinalHyperparameter(name=param_name, sequence=param_values)
            # We are throwing away information here, but OrderedDiscrete
            # appears to be invariant to monotonic transformation anyway.
        elif param_type == "int":
            assert param_values is None
            # Need +1 since API in inclusive
            # choices = range(int(param_range[0]), int(param_range[-1]) + 1)

            arg = CSH.UniformIntegerHyperparameter(name=param_name,
                                                   lower=param_range[0],
                                                   upper=param_range[1],
                                                   log=param_space == 'log'
                                                   )
            # We are throwing away information here, but OrderedDiscrete
            # appears to be invariant to monotonic transformation anyway.
        elif param_type == "real":
            assert param_values is None
            assert param_range is not None
            # Will need to warp to this space sep.

            if param_space in ('linear', 'log'):
                arg = CSH.UniformFloatHyperparameter(name=param_name,
                                                   lower=param_range[0],
                                                   upper=param_range[1],
                                                   log=param_space == 'log'
                                                   )
            else:
                # raise NotImplementedError()
                range_warped = cs.warp_space(param_range, warp=param_space)
                arg = CSH.UniformFloatHyperparameter(name=param_name,
                                                   lower=range_warped[0],
                                                   upper=range_warped[1],
                                                   )
                prewarp = param_space
                print(param_space)
        else:
            assert False, "type %s not handled in API" % param_type
        cs.add_hyperparameter(arg)
        cs.all_warp[param_name] = prewarp
        # all_args[param_name] = arg
        # all_prewarp[param_name] = prewarp


    # config_spaces =
    return cs