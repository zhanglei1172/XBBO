import logging
logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_space.fast_example_problem import rosenbrock_2d
from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.search_algorithm import alg_register




def run_one_exp(opt_name, max_call, seed):
     # Build Configuration Space which defines all parameters and their ranges
    cs = DenseConfigurationSpace(seed)
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])

    hpopt = alg_register[opt_name](space=cs, seed=seed, total_limit=max_call, initial_design='sobol',)

    # ---- Begin BO-loop ----
    for i in range(max_call):
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
        best_val = run_one_exp('tpe', 50, rng.randint(1e5))
        best_vals.append(best_val)
    print(best_vals)