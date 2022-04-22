import logging

logging.basicConfig(level=logging.INFO)

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_space.fast_example_problem import branin
# from xbbo.configspace.space import ConfigurationSpace
from xbbo.search_algorithm import alg_register


def run_one_exp(opt_name, max_call, seed):
    option_kwargs = {
        'turbo-1': {
            'name': 'turbo',
            'kwargs': {
                'num_tr': 1
            }
        },
        'turbo-2': {
            'name': 'turbo',
            'kwargs': {
                'num_tr': 2
            }
        },
        # 'rea':{
        #     'name':'rea',
        #     'kwargs':{
        #         'llambda': 100,
        #         # 'sample_size': 30
        #     }
        # }
    }
    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace(seed=seed)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])
    if opt_name in option_kwargs:
        dic = option_kwargs[opt_name]
        hpopt = alg_register[dic['name']](space=cs,
                                          seed=seed,
                                          suggest_limit=max_call,
                                          **dic['kwargs'])
    else:
        hpopt = alg_register[opt_name](
            space=cs,
            seed=seed,
            suggest_limit=max_call,
        )

    # ---- Begin BO-loop ----
    for i in range(max_call):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate
        for trial in trial_list:
            value = branin(trial.config_dict)
            # observe
            trial.add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)

        print(value)
    losses = np.array(hpopt.trials._his_observe_value)
    # res = {'optimal': losses.min(), 'opt_call': losses.argmin() + 1}
    return losses  # np.minimum.accumulate(hpopt.trials.get_history()[0])


def benchmark(test_algs,
              func_to_call,
              max_call=200,
              repeat_num=10,
              father_seed=42,
              desc=''):
    import prettytable as pt

    results_all = []
    for test_alg in test_algs:
        rng = np.random.RandomState(father_seed)
        results_ = []
        for _ in range(repeat_num):
            seed = rng.randint(1e5)
            losses = func_to_call(test_alg, max_call, seed)
            res = [losses.min(), losses.argmin() + 1]

            results_.append(res)
        results_all.append(results_)
    results_all = np.array(results_all)  # (alg, repeat, 2)
    print(results_all)

    tb = pt.PrettyTable([
        "Method", "Minimum", "Best minimum", "Mean f_calls to min",
        "Std f_calls to min", "Fastest f_calls to min"
    ])
    # tb = pt.PrettyTable()
    # tb.field_names = []
    for i, results in enumerate(results_all):
        dict_result = {
            "mean": np.round(results.mean(axis=0), 3),
            "std": np.round(results.std(axis=0), 3),
            "best": np.round(results.min(axis=0), 3)
        }
        tb.add_row([
            '{}({})'.format(desc, test_algs[i]),
            '{:.3f}+/-{:.3f}'.format(dict_result["mean"][0],
                                     dict_result['std'][0]),
            dict_result["best"][0], dict_result["mean"][1],
            dict_result["std"][1],
            int(dict_result["best"][1])
        ])

    print(tb)
    return results_all


if __name__ == "__main__":
    # bore currently has some bugs
    test_algs = [
        'anneal', 'basic-bo', 'tpe', 'cem', 'cma-es', 'de', 'rs', 'rea',
        'turbo-1',
        'turbo-2',
        'bore'
    ]  # 'nsga2','bo-transfer','pbt'
    # benchmark(test_algs, run_one_exp, 200, 10, 42, desc='XBBO')
    benchmark(test_algs, run_one_exp, 20, 1, 42, desc='XBBO') # for fast test