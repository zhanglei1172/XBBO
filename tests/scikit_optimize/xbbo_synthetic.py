import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_space.fast_example_problem import branin
from xbbo.configspace.space import DenseConfigurationSpace
from xbbo.search_algorithm import alg_register


def run_one_exp(opt_name, max_call, seed):
    # Build Configuration Space which defines all parameters and their ranges
    cs = DenseConfigurationSpace(seed)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])

    hpopt = alg_register[opt_name](
        space=cs,
        seed=seed,
        total_limit=max_call,
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
    losses = np.round(hpopt.trials._his_observe_value, 3)
    # res = {'optimal': losses.min(), 'opt_call': losses.argmin() + 1}
    return losses.min(), losses.argmin(
    ) + 1  # np.minimum.accumulate(hpopt.trials.get_history()[0])


if __name__ == "__main__":
    import prettytable as pt

    repeat_num = 10
    max_call = 200
    test_alg = 'basic-bo'
    rng = np.random.RandomState(42)

    results = []
    for _ in range(repeat_num):
        res = run_one_exp(test_alg, max_call, rng.randint(1e5))

        results.append(res)

    results = np.array(results)
    print(results)

    tb = pt.PrettyTable([
        "Method", "Minimum", "Best minimum", "Mean f_calls to min",
        "Std f_calls to min", "Fastest f_calls to min"
    ])
    # tb = pt.PrettyTable()
    # tb.field_names = []
    dict_result = {
        "mean": results.mean(axis=0),
        "std": results.std(axis=0),
        "best": results.min(axis=0)
    }
    tb.add_row([
        "XBBO-bo-gp", '{:.3f}+/-{:.3f}'.format(dict_result["mean"][0],
                                               dict_result['std'][0]),
        dict_result["best"][0], dict_result["mean"][1], dict_result["std"][1],
        dict_result["best"][1]
    ])

    print(tb)
