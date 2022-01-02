import numpy as np
# import matplotlib.pyplot as plt
from xbbo.configspace.space import DenseConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_algorithm.de_optimizer import DE

def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """

    x1 = x["x0"]
    x2 = x["x1"]

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y

def build_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])
    return cs

def build_branin_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])
    return cs

if __name__ == "__main__":
    MAX_CALL = 100
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = rosenbrock_2d
    # define search space
    cs = build_space(rng)
    # define black box optimizer
    hpopt = DE(space=cs, seed=rng.randint(10000),llambda=10)
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

