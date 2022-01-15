import numpy as np
import argparse

from skopt.benchmarks import branin
from skopt import gp_minimize
from skopt import forest_minimize
from skopt import gbrt_minimize
from skopt import dummy_minimize


def run(n_calls, n_runs, optimizers, acq_optimizer="lbfgs"):
    bounds = [(-5.0, 10.0), (0.0, 15.0)]

    results = []
    for name, optimizer in optimizers:
        print(name)
        results_ = []
        min_func_calls = []
        time_ = 0.0

        for random_state in range(n_runs):
            if name == "gp_minimize":
                res = optimizer(branin,
                                bounds,
                                random_state=random_state,
                                n_calls=n_calls,
                                noise=1e-10,
                                verbose=True,
                                acq_optimizer=acq_optimizer,
                                n_jobs=-1)
            elif name == "gbrt_minimize":
                res = optimizer(branin,
                                bounds,
                                random_state=random_state,
                                n_calls=n_calls,
                                acq_optimizer=acq_optimizer)
            else:
                res = optimizer(branin,
                                bounds,
                                random_state=random_state,
                                n_calls=n_calls,)
            results_.append(res)
            func_vals = np.round(res.func_vals, 3)
            min_func_calls.append(np.argmin(func_vals) + 1)

        optimal_values = [result.fun for result in results_]
        mean_optimum = np.mean(optimal_values)
        std = np.std(optimal_values)
        best = np.min(optimal_values)
        print("Mean optimum: " + str(mean_optimum))
        print("Std of optimal values" + str(std))
        print("Best optima:" + str(best))

        mean_fcalls = np.mean(min_func_calls)
        std_fcalls = np.std(min_func_calls)
        best_fcalls = np.min(min_func_calls)
        print("Mean func_calls to reach min: " + str(mean_fcalls))
        print("Std func_calls to reach min: " + str(std_fcalls))
        print("Fastest no of func_calls to reach min: " + str(best_fcalls))

        results.append([optimal_values, min_func_calls])
    return np.swapaxes(np.array(results), 1, 2)  # (alg, run, 2)


if __name__ == "__main__":
    repeat_num = 10
    max_call = 200
    optimizers = [("gp_minimize", gp_minimize),
                  ("forest_minimize", forest_minimize),
                  ("gbrt_minimize", gbrt_minimize),
                  ("dummy_minimize", dummy_minimize)]
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     '--n_calls', nargs="?", default=200, type=int, help="Number of function calls.")
    # parser.add_argument(
    #     '--n_runs', nargs="?", default=10, type=int, help="Number of runs.")
    # parser.add_argument(
    #     '--acq_optimizer', nargs="?", default="lbfgs", type=str,
    #     help="Acquistion optimizer.")
    # args = parser.parse_args()
    # run(args.n_calls, args.n_runs, args.acq_optimizer)
    import prettytable as pt
    results_all = run(max_call, repeat_num, optimizers)
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
            'skopt({})'.format(optimizers[i][0]), '{:.3f}+/-{:.3f}'.format(dict_result["mean"][0],
                                                   dict_result['std'][0]),
            dict_result["best"][0], dict_result["mean"][1],
            dict_result["std"][1], int(dict_result["best"][1])
        ])

    print(tb)