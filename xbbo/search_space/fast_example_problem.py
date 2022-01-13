import numpy as np
from xbbo.configspace.space import DenseConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter, CategoricalHyperparameter
from ConfigSpace.conditions import LessThanCondition

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
    # x2 = x.get('x1', x1)

    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val

def branin(config):
    x1, x2 = config['x1'], config['x2']
    y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10
    return y

def rosenbrock_2d_hard(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """

    x1 = x["x0"]
    # x2 = x["x1"]
    x2 = x.get('x1', x1)
    x3 = x['x2']
    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val - (x3 == 2)

def build_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])
    # con = LessThanCondition(x1, x0, 1.)
    # cs.add_condition(con)
    return cs

def build_branin_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])
    return cs


def build_space_hard(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x0 = UniformFloatHyperparameter("x0", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=-4)
    x2 = CategoricalHyperparameter("x2", choices=[0,1,2,3])
    cs.add_hyperparameters([x0, x1, x2])
    con = LessThanCondition(x1, x0, 1.)
    cs.add_condition(con)
    return cs


def zdt1(config):
    n = 30
    x1, x2 = config['x1'], config['x2']
    sigma = x2 # sum(input_x[1:])
    g = 1 + sigma * 9 / (n - 1)
    h = 1 - (x1 / g)**0.5
    return x1, g*h


def build_zdt1_space(rng):
    cs = DenseConfigurationSpace(seed=rng.randint(10000))
    x0 = UniformFloatHyperparameter("x1", 0, 1)
    x1 = UniformFloatHyperparameter("x2", 0, 1)
    cs.add_hyperparameters([x0, x1])
    return cs
