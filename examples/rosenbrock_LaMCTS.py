import numpy as np
# import matplotlib.pyplot as plt
from xbbo.search_space.fast_example_problem import Ackley

from xbbo.search_algorithm.lamcts import LaMCTS
from xbbo.utils.constants import MAXINT



if __name__ == "__main__":
    MAX_CALL = 1000
    rng = np.random.RandomState(42)
    problem = Ackley(20, np.random.RandomState(0))
    # define black box function
    blackbox_func = problem.objective_function
    # define search space
    cs = problem.get_configuration_space()
    # define black box optimizer
    hpopt = LaMCTS(space=cs,objective_function=blackbox_func, suggest_limit=MAX_CALL, seed=rng.randint(MAXINT), C_p=1, leaf_size=10, init_budget=40, kernel_type='rbf', gamma_type='auto',verbose=True,split_metric='mean')
    # hpopt = LaMCTS(space=cs, seed=rng.randint(MAXINT), C_p=1, leaf_size=10, init_budget=40, kernel_type='rbf', gamma_type='auto',verbose=True, split_metric='mean', solver='random')
    
    # Example call of the black-box function
    def_value = blackbox_func(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest()
        # evaluate 
        value = blackbox_func(trial_list[0].config_dict)
        # observe
        trial_list[0].add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)
        
        print(value)
    # hpopt.optimize()
    # best_value, best_config = hpopt.trials.get_best()
    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_gp.png')
    # plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()[0]))
