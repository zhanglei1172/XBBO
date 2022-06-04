import numpy as np
from xbbo.problem.fast_example_problem import Rosenbrock
from xbbo.search_algorithm.bore_optimizer import BORE
from xbbo.core.constants import MAXINT

if __name__ == "__main__":
    MAX_CALL = 200
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = Rosenbrock(rng=rng)
    # define search space
    cs = blackbox_func.get_configuration_space()
    # define black box optimizer
    hpopt = BORE(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol', classify='rf')
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate 
        obs = blackbox_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(obs)
        hpopt.observe(trial_list=trial_list)
        
        print(obs)
    
    print('find best (value, config):{}'.format(hpopt.trials.get_best()))

