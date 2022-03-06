import numpy as np
import matplotlib.pyplot as plt

from xbbo.search_algorithm.nsga_optimizer import NSGAII
from xbbo.search_space.fast_example_problem import build_zdt1_space, zdt1
from xbbo.utils.constants import MAXINT



if __name__ == "__main__":
    MAX_CALL = 1000
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = zdt1
    # define search space
    cs = build_zdt1_space(rng)
    # define black box optimizer
    hpopt = NSGAII(space=cs, seed=rng.randint(MAXINT),llambda=30)
    # Example call of the black-box function
    def_value = blackbox_func(cs.get_default_configuration())
    print("Default Value:{}".format(def_value))
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
    points = np.asarray(hpopt.trials.get_history()[0])
    plt.scatter(points[-100:,0], points[-100:, 1], s=10, marker='o')
    plt.axis('equal')
    plt.savefig('./out/zdt1_nsga2.png')

