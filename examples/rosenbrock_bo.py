import numpy as np

from xbbo.search_space.fast_example_problem import build_space_hard, rosenbrock_2d_hard
from xbbo.search_algorithm.bo_optimizer import BO

if __name__ == "__main__":
    MAX_CALL = 30
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = rosenbrock_2d_hard
    # define search space
    cs = build_space_hard(rng)
    # define black box optimizer
    hpopt = BO(space=cs, seed=rng.randint(10000), total_limit=MAX_CALL, initial_design='sobol', surrogate='gp', acq_opt='rs_ls')
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
    # plt.savefig('./out/rosenbrock_bo_gp.png')
    # plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()[0]))
