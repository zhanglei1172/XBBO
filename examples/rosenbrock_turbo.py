import numpy as np
# import matplotlib.pyplot as plt
from xbbo.search_space.fast_example_problem import build_branin_space, branin

from xbbo.search_algorithm.turbo_optimizer import TuRBO
from xbbo.utils.constants import MAXINT


if __name__ == "__main__":
    MAX_CALL = 20
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = branin
    # define search space
    cs = build_branin_space(rng)
    # define black box optimizer
    hpopt = TuRBO(space=cs, seed=rng.randint(MAXINT), initial_design='sobol', num_tr=1)
    # Example call of the black-box function
    def_value = blackbox_func(cs.get_default_configuration())
    print("Default Value: %.2f" % def_value)
    # ---- Begin BO-loop ----
    for i in range(MAX_CALL):
        # suggest
        trial_list = hpopt.suggest(10)
        # evaluate 
        for trial in trial_list:
            value = blackbox_func(trial.config_dict)
            # observe
            trial.add_observe_value(observe_value=value)
        hpopt.observe(trial_list=trial_list)
        
        print(value)
    
    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_rs.png')
    # plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()[0]))

