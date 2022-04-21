import logging

from comparison.xbbo_benchmark import benchmark

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_space.fast_example_problem import branin
from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.search_algorithm import alg_register


def run_one_exp(opt_name, max_call, seed):
    # Build Configuration Space which defines all parameters and their ranges
    cs = DenseConfigurationSpace(seed=seed)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])

    hpopt = alg_register[opt_name](
        space=cs,
        seed=seed,
        suggest_limit=max_call,
        initial_design='sobol',
    )

    # ---- Begin BO-loop ----
    for i in range(max_call):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate
        value = branin(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)

        print(value)

    return np.array(hpopt.trials.get_history()[0])


if __name__ == "__main__":
    test_algs = ['tpe', 'anneal', 'rs']  # 'nsga2','bo-transfer','pbt'
    benchmark(test_algs, run_one_exp, 200, 10, 42, desc='XBBO')
    # rng = np.random.RandomState(42)
    # best_vals = []
    # for _ in range(3):
    #     losses = run_one_exp('tpe', 50, rng.randint(1e5))
    #     best_val = np.minimum.accumulate(losses)
    #     best_vals.append(best_val)
    # print(best_vals)