import numpy as np
import matplotlib.pyplot as plt

from xbbo.search_algorithm.multi_obj.nsga_optimizer import NSGAII
from xbbo.problem.fast_example_problem import ZDT1
from xbbo.core.constants import MAXINT


if __name__ == "__main__":
    MAX_CALL = 1000
    rng = np.random.RandomState(0)

    # define black box function
    mo_blackbox_func = ZDT1(rng=rng)
    # define search space
    cs = mo_blackbox_func.get_configuration_space()
    # define black box optimizer
    hpopt = NSGAII(space=cs, seed=rng.randint(MAXINT),llambda=30)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate 
        obs = mo_blackbox_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(obs)
        hpopt.observe(trial_list=trial_list)
        
        print(obs)
    

    print('find best (value, config):{}'.format(hpopt.trials.get_best()))

