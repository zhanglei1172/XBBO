import ConfigSpace
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np

CS.Configuration

def value2range(dtype, values=None, range_=None):
    assert (values is None) != (range_ is None)
    round_to_values = default_round
    if range_ is None:  # => value is not None
        # Debatable if unique should be done before or after cast. But I
        # think after is better, esp. when changing precisions.
        values = np.asarray(values, dtype=dtype)
        values = np.unique(values)  # values now 1D ndarray no matter what

        # Extrapolation might happen due to numerics in type conversions.
        # Bounds checking is still done in validate routines.
        # round_to_values = interp1d(values, values, kind="nearest", fill_value="extrapolate")
        range_ = (values[0], values[-1])
    # Save values and rounding
    # Values is either None or was validated inside if statement

    # Note that if dtype=None that is the default for asarray.
    range_ = np.asarray(range_, dtype=dtype)

    # Save range info, with input validation and post validation
    return range_


def _real(name, warp="linear", values=None, range_=None):
    assert values is None, "cannot pass in values for real"
    assert range_ is not None, "must pass in explicit range for real"
    # lower, upper = value2range(np.float_, values, range_)
    if warp == "log":
        return CSH.UniformFloatHyperparameter(name=name, lower=10, upper=100, log=True), None
    return CSH.UniformFloatHyperparameter(name=name, lower=10, upper=100), warp

def _int(name, warp="linear", values=None, range_=None):
    lower, upper = value2range(np.float_, values, range_)

    return CSH.UniformIntegerHyperparameter()

def _cat(name, warp="linear", values=None, range_=None):
    assert warp is None, "cannot warp cat"
    # assert values is not None, "must pass in explicit values for cat"
    assert range_ is None, "cannot pass in range for cat"
    # values = np.unique(values)
    # lower, upper = value2range(np.float_, values, range_)

    return CSH.CategoricalHyperparameter(name=name, choices=values), None

def _ordinal(name, warp="linear", values=None, range_=None):
    lower, upper = value2range(np.float_, values, range_)

    return CSH.OrdinalHyperparameter()

def _bool(name, warp="linear", values=None, range_=None):
    assert warp is None, "cannot warp bool"
    assert values is None, "cannot pass in values for bool"
    assert range_ is None, "cannot pass in range for bool"
    # lower, upper = value2range(np.float_, values, range_)

    return CSH.OrdinalHyperparameter(name=name, sequence=[False, True]), None

SPACE_DICT = {
    "real": _real,
    "int": _int,
    "bool": _bool,
    "cat": _cat,
    "ordinal": _ordinal
}

def build_space(meta):
    """Build Real space class.

    Parameters
    ----------
    meta : dict(str, dict)
        Configuration of variables in joint space. See API description.
    """
    cs = CS.ConfigurationSpace(seed=1234)
    param_list = sorted(meta.keys())
    for param_name in param_list:
        param_config = meta[param_name]

        param_type = param_config["type"]

        param_space = param_config.get("space", None)
        param_range = param_config.get("range", None)
        param_values = param_config.get("values", None)

        prewarp = None
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
                return NotImplementedError()
                # prewarp = Real(warp=param_space, range_=param_range)
        else:
            assert False, "type %s not handled in API" % param_type
        cs.add_hyperparameter(arg)
        # all_args[param_name] = arg
        # all_prewarp[param_name] = prewarp


    # config_spaces =
    return cs

class JointSpace(object):
    """Combination of multiple :class:`.Space` objectives to transform multiple variables at the same time (jointly).
    """

    def __init__(self, meta):
        """Build Real space class.

        Parameters
        ----------
        meta : dict(str, dict)
            Configuration of variables in joint space. See API description.
        """
        assert len(meta) > 0  # Unclear what to do with empty space
        cs = CS.ConfigurationSpace(seed=1234)
        # Lock in an order if not ordered dict, sorted helps reproducibility
        self.param_list = sorted(meta.keys())

        # Might as well pre-validate a bit here
        for param, config in meta.items():
            assert config["type"] in SPACE_DICT, "invalid input type %s" % config["type"]

        spaces = {
            param: SPACE_DICT[config["type"]](
                config.get("space", None), config.get("values", None), config.get("range", None)
            )
            for param, config in meta.items()
        }
        self.spaces = spaces

        self.blocks = np.cumsum([len(spaces[param].get_bounds()) for param in self.param_list])