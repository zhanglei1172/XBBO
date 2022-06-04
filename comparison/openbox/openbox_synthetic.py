# Import Openbox Library
import sys
sys.path.append("/home/leizhang/project/open-box/")
import numpy as np
import matplotlib.pyplot as plt
from openbox import Advisor, sp, Observation

from xbbo.problem.fast_example_problem import branin

import numpy as np


def run_one_exp(opt_name, max_call, seed):
    space = sp.Space()
    x1 = sp.Real("x1", -5, 10, default_value=0)
    x2 = sp.Real("x2", 0, 15, default_value=0)
    space.add_variables([x1, x2])
    advisor = Advisor(
        space,
        # surrogate_type='gp',
        surrogate_type=opt_name,
        task_id='quick_start',
        random_state=np.random.RandomState(seed)
    )

    MAX_RUNS = max_call
    for i in range(MAX_RUNS):
        # ask
        config = advisor.get_suggestion()
        # evaluate
        ret = branin(config)
        # tell
        observation = Observation(config=config, objs=[ret])
        advisor.update_observation(observation)
        print('===== ITER %d/%d: %s.' % (i+1, MAX_RUNS, observation))

    history = advisor.get_history()
    # print(best)
    losses = np.asarray(history.perfs)
    # return np.minimum.accumulate(losses)
    return losses


if __name__ == "__main__":
    from comparison.xbbo_benchmark import benchmark
    benchmark(['auto'], run_one_exp, 200, 10, 42, desc='openbox')
    # rng = np.random.RandomState(42)
    # best_vals = []
    # for _ in range(3):
    #     best_val = run_one_exp('tpe', 50, rng.randint(1e5))
    #     best_vals.append(best_val)
    # print(best_vals)