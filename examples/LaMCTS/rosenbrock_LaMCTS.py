import numpy as np
# import matplotlib.pyplot as plt
from xbbo.problem.fast_example_problem import Ackley

from xbbo.search_algorithm.lamcts import LaMCTS
from xbbo.core.constants import MAXINT



if __name__ == "__main__":
    MAX_CALL = 1000
    rng = np.random.RandomState(42)
    # define black box function
    blackbox_func = Ackley(20, rng=rng)
    # define search space
    cs = blackbox_func.get_configuration_space()
    # define black box optimizer
    hpopt = LaMCTS(space=cs,objective_function=blackbox_func, suggest_limit=MAX_CALL, seed=rng.randint(MAXINT), C_p=1, leaf_size=10, init_budget=40, kernel_type='rbf', gamma_type='auto',verbose=True,split_metric='mean', split_use_predict=True)
    # hpopt = LaMCTS(space=cs, seed=rng.randint(MAXINT), C_p=1, leaf_size=10, init_budget=40, kernel_type='rbf', gamma_type='auto',verbose=True, split_metric='mean', solver='random')
    
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
        print('Iter {} :  Find best value:{}'.format(i, hpopt.trials.get_best()[0])) 

    print('find best (value, config):{}'.format(hpopt.trials.get_best()))

