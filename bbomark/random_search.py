import numpy as np

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def suggest_dict(X, y, config_space, n_suggestions=1):
    '''

    :param X:
    :param y:
    :param config_space:
    :param n_suggestions:
    :param random:
    :return: List[param dict form]
    '''

    config_randoms = config_space.sample_configuration(size=n_suggestions)
    if n_suggestions == 1:
        return [config_randoms.get_dictionary()]
    next_guess = [config_random.get_dictionary() for config_random in config_randoms]
    # space_x = JointSpace(meta)
    # X_warped = space_x.warp(X)
    # bounds = space_x.get_bounds()
    # _, n_params = _check_x_y(X_warped, y, allow_impute=True)
    # lb, ub = _check_bounds(bounds, n_params)
    #
    # # Get the suggestion
    # suggest_x = random.uniform(lb, ub, size=(n_suggestions, n_params))
    #
    # # Unwarp
    # next_guess = space_x.unwarp(suggest_x)
    return next_guess

