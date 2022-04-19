# Import HyperOpt Library
import sys
sys.path.append('../hypermapper/')
from hypermapper import optimizer  # noqa 
import json, os, math
from xbbo.search_space.fast_example_problem import branin
import pandas as pd

import numpy as np

opt_name_map = {
    'rs': "random_scalarizations",
    'bo': "bayesian_optimization",
    'local_search':"local_search",
    'evolution':"evolutionary_optimization"
    
}
def branin_function(X):
    """
    Compute the branin function.
    :param X: dictionary containing the input points.
    :return: the value of the branin function
    """
    values = []
    if not np.iterable((X["x1"])):
        X["x1"] = [X["x1"]]
        X["x2"] = [X["x2"]]
    for idx in range(len(X["x1"])):
        x1 = X["x1"][idx]
        x2 = X["x2"][idx]
        a = 1.0
        b = 5.1 / (4.0 * math.pi * math.pi)
        c = 5.0 / math.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * math.pi)

        y_value = (
            a * (x2 - b * x1 * x1 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
        )
        values.append(y_value)

    return values

def run_one_exp(opt_name, max_call, seed):
    # Define the search space of x between -10 and 10.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'branin_scenario.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    config['optimization_method'] = opt_name_map[opt_name]
    config['optimization_iterations'] = max_call
    with open(config_path, 'w') as f:
        json.dump(config,f)
    state = np.random.get_state()
    np.random.seed(seed)
    optimizer.optimize(config_path, branin_function)
    # print(best)
    df = pd.read_csv('./branin_output_samples.csv')
    losses = df['Value'].values
    np.random.set_state(state)
    # return np.minimum.accumulate(losses)
    return losses


if __name__ == "__main__":
    from comparison.xbbo_benchmark import benchmark
    benchmark(['bo', 'evolution'], run_one_exp, 200, 10, 42, desc='hypermapper')
    # rng = np.random.RandomState(42)
    # best_vals = []
    # for _ in range(3):
    #     best_val = run_one_exp('tpe', 50, rng.randint(1e5))
    #     best_vals.append(best_val)
    # print(best_vals)