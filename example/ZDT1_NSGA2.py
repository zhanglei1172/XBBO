import numpy as np
import matplotlib.pyplot as plt
from xbbo.configspace.space import DenseConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_algorithm.nsga_optimizer import NSGAII


def zdt1(config):
    n = 30
    x1, x2 = config['x1'], config['x2']
    sigma = x2 # sum(input_x[1:])
    g = 1 + sigma * 9 / (n - 1)
    h = 1 - (x1 / g)**0.5
    return x1, g*h


def build_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x0 = UniformFloatHyperparameter("x1", 0, 1)
    x1 = UniformFloatHyperparameter("x2", 0, 1)
    cs.add_hyperparameters([x0, x1])
    return cs

if __name__ == "__main__":
    MAX_CALL = 1000
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = zdt1
    # define search space
    cs = build_space(rng)
    # define black box optimizer
    hpopt = NSGAII(space=cs, seed=rng.randint(10000),llambda=30)
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

