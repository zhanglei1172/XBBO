import numpy as np
# import matplotlib.pyplot as plt
from xbbo.search_space.fast_example_problem import build_space, rosenbrock_2d

from xbbo.search_algorithm.turbo_optimizer import TuRBO


if __name__ == "__main__":
    MAX_CALL = 30
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = rosenbrock_2d
    # define search space
    cs = build_space(rng)
    # define black box optimizer
    hpopt = TuRBO(space=cs, seed=rng.randint(10000), initial_design='sobol', num_tr=1)
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
    
    # plt.plot(hpopt.trials.get_history()[0])
    # plt.savefig('./out/rosenbrock_bo_rs.png')
    # plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()[0]))

