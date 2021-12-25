import numpy as np
import matplotlib.pyplot as plt
from xbbo.configspace.space import DenseConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from xbbo.search_algorithm.bo_gp_optimizer import BOGP

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

def build_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])
    return cs

if __name__ == "__main__":
    MAX_CALL = 30
    rng = np.random.RandomState(42)

    # define black box function
    blackbox_func = rosenbrock_2d
    # define search space
    cs = build_space(rng)
    # define black box optimizer
    hpopt = BOGP(config_spaces=cs, seed=rng.randint(10000), total_limit=MAX_CALL)
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
    
    plt.plot(np.log(hpopt.trials.get_history()[0]))
    plt.savefig('./out/rosenbrock_bo_gp.png')
    plt.show()
    print('find best value:{}'.format(hpopt.trials.get_best()[0]))

