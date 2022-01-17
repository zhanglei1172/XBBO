import sys
sys.path.append('/home/leizhang/project/BayesianOptimization/')

import numpy as np
from bayes_opt import BayesianOptimization


def branin(x1, x2):
    # x1, x2 = x['x1'], x['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return -y


alg_map = {
    'bo': 1,
}


def run_one_exp(opt_name, max_call, seed):
    pbounds = {'x1': (-5, 10), 'x2': (10, 15)}

    optimizer = BayesianOptimization(f=branin,
                                     pbounds=pbounds,
                                     verbose=2,
                                     random_state=seed,)
    optimizer.maximize(
        init_points=16,
        n_iter=max_call-16,
    )

    losses = -np.array(optimizer.space.target)
    # return np.minimum.accumulate(losses)
    return losses


if __name__ == "__main__":
    from tests.xbbo_benchmark import benchmark
    benchmark(list(alg_map.keys()), run_one_exp, 200, 10, 42, desc='bayes_opt')
    # rng = np.random.RandomState(42)
    # best_vals = []
    # for _ in range(3):
    #     best_val = run_one_exp('tpe', 50, rng.randint(1e5))
    #     best_vals.append(best_val)
    # print(best_vals)