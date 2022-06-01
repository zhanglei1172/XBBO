import numpy as np

from xbbo.problem.fast_example_problem import Rosenbrock
from xbbo.search_algorithm.anneal_optimizer import Anneal
from xbbo.core.constants import MAXINT


if __name__ == "__main__":
    MAX_CALL = 1000
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = Rosenbrock(rng=rng)
    # define search space
    cs = blackbox_func.get_configuration_space()
    # define black box optimizer
    hpopt = Anneal(space=cs, seed=rng.randint(MAXINT), suggest_limit=MAX_CALL, initial_design='sobol',init_budget=1)
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
    
    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_gp.png')
    # plt.show()
    print('find best (value, config):{}'.format(hpopt.trials.get_best()))

