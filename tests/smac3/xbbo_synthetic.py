import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.search_algorithm.bo_optimizer import BO

MAX_CALL = 30



def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """

    x1 = x["x0"]
    x2 = x["x1"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

def run_one_exp(seed):
     # Build Configuration Space which defines all parameters and their ranges
    cs = DenseConfigurationSpace(seed)
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    hpopt = BO(space=cs,
               seed=seed,
               total_limit=MAX_CALL,
               initial_design='sobol',
               surrogate='gp',
               acq_opt='rs_ls')

    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate
        value = rosenbrock_2d(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)

        print(value)

    return np.minimum.accumulate(hpopt.trials.get_history()[0])


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    best_vals = []
    for _ in range(3):
        best_val = run_one_exp(rng.randint(1e5))
        best_vals.append(best_val)
    print(best_vals)