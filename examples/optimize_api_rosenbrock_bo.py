import numpy as np

from xbbo.search_space.fast_example_problem import build_space_hard, rosenbrock_2d_hard
from xbbo.search_algorithm.bo_optimizer import BO
from xbbo.utils.constants import MAXINT

if __name__ == "__main__":
    MAX_CALL = 30
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = rosenbrock_2d_hard
    # define search space
    cs = build_space_hard(rng)
    # define black box optimizer
    hpopt = BO(space=cs,
               objective_function=blackbox_func,
               seed=rng.randint(MAXINT),
               suggest_limit=MAX_CALL,
               initial_design='sobol',
               surrogate='gp',
               acq_opt='rs_ls')

    # ---- Use minimize API ----
    hpopt.optimize()
    best_value, best_config = hpopt.trials.get_best()
    print('Find best value:{}'.format(best_value))
    print('Best Config:{}'.format(best_config))
