# Import HyperOpt Library
from hyperopt import tpe, atpe, mix, anneal, rand, hp, fmin, Trials

from xbbo.search_space.fast_example_problem import rosenbrock_2d

import numpy as np

opt_name_map = {
    'tpe': tpe,
    'rand': rand,
    'mix': mix,
    'atpe': atpe,
    'anneal': anneal
}

def run_one_exp(opt_name, max_call, seed):
    # Define the search space of x between -10 and 10.
    space = space = {
        'x0': hp.uniform('x0', -5, 10),
        'x1': hp.uniform('x1', -5, 10)
    }
    trials = Trials()
    best = fmin(
        fn=rosenbrock_2d, # Objective Function to optimize
        space=space, # Hyperparameter's Search Space
        algo=opt_name_map[opt_name].suggest, # Optimization algorithm
        # max_evals=1000, # Number of optimization attempts
        max_evals=max_call,
        trials=trials,
        rstate=np.random.RandomState(seed)
    )
    # print(best)
    losses = np.asarray(trials.losses())
    return np.minimum.accumulate(losses)


if __name__ == "__main__":
    rng = np.random.RandomState(42)
    best_vals = []
    for _ in range(3):
        best_val = run_one_exp('tpe',50, rng.randint(1e5))
        best_vals.append(best_val)
    print(best_vals)