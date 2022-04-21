import numpy as np
import time
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from ConfigSpace import ConfigurationSpace

from ConfigSpace.conditions import InCondition, LessThanCondition
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

from xbbo.utils.constants import MAXINT, Key


def mf_stochastic_count_one(config, info=None):
    '''
    $$
    f(x)=-\left(\sum_{x \in X_{c a t}} x+\sum_{x \in X_{c o n t}} b\left[\left(B_{p=x}\right)\right]\right)
    $$
    '''
    if info is None:
        info = {}
    budget = info.get(Key.BUDGET, 100)
    random_state = info.get('random_state', np.random.RandomState(0))
    xs = []
    rs = []
    st = time.time()
    for k in config:
        if k[0] == "x":
            xs.append(config[k])
        else:
            rs.append(random_state.binomial(1, config[k], size=int(budget)).mean())
    xs = -np.array(xs).sum()
    rs = -np.array(rs).sum()

    # result dict passed to DE/DEHB as function evaluation output
    res = {
        "fitness": xs+rs,  # must-have key that DE/DEHB minimizes
        # "cost": budget,  # must-have key that associates cost/runtime 
        Key.EVAL_TIME: time.time() - st
        # "info": dict() # optional key containing a dictionary of additional info
    }
    res.update(info)
    # dict representation that DEHB requires
    # res = {
    #     "fitness": loss,
    #     "cost": cost,
    #     "info": {"test_loss": test_loss, Key.BUDGET: budget}
    # }
    return res

def build_mf_SCO_space(rng, dim=8):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    xs = [CategoricalHyperparameter('x{}'.format(i), choices=[0,1]) for i in range(dim)]
    ys = [UniformFloatHyperparameter('y{}'.format(i), 0, 1, default_value=0.5) for i in range(dim)]
    cs.add_hyperparameters(xs)
    cs.add_hyperparameters(ys)
    return cs


def rosenbrock_2d(x):
    """ The 2 dimensional Rosenbrock function as a toy model
    The Rosenbrock function is well know in the optimization community and
    often serves as a toy problem. It can be defined for arbitrary
    dimensions. The minimium is always at x_i = 1 with a function value of
    zero. All input parameters are continuous. The search domain for
    all x's is the interval [-5, 10].
    """

    x1 = x["x1"]
    x2 = x["x2"]
    # x2 = x.get('x2', x1)

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

    x1 = x["x1"]
    # x2 = x["x1"]
    x2 = x.get('x2', x1)
    x3 = x['x3']
    val = 100. * (x2 - x1 ** 2.) ** 2. + (1 - x1) ** 2.
    return val - (x3 == 2)

def build_space(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x0 = UniformFloatHyperparameter("x1", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x2", -5, 10, default_value=-4)
    cs.add_hyperparameters([x0, x1])
    # con = LessThanCondition(x1, x0, 1.)
    # cs.add_condition(con)
    return cs

def build_branin_space(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
    x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
    cs.add_hyperparameters([x1, x2])
    return cs


def build_space_hard(rng):
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x0 = UniformFloatHyperparameter("x1", -5, 10, default_value=-3)
    x1 = UniformFloatHyperparameter("x2", -5, 10, default_value=-4)
    x2 = CategoricalHyperparameter("x3", choices=[0,1,2,3])
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
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))
    x0 = UniformFloatHyperparameter("x1", 0, 1)
    x1 = UniformFloatHyperparameter("x2", 0, 1)
    cs.add_hyperparameters([x0, x1])
    return cs

def svm_from_cfg(cfg):
    """ Creates a SVM based on a configuration and evaluates it on the
    iris-dataset using cross-validation. Note here random seed is fixed

    Parameters:
    -----------
    cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
        Configuration containing the parameters.
        Configurations are indexable!

    Returns:
    --------
    A crossvalidated mean score for the svm on the loaded data-set.
    """
    # For deactivated parameters, the configuration stores None-values.
    # This is not accepted by the SVM, so we remove them.
    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    # And for gamma, we set it to a fixed value or to "auto" (if used)
    if "gamma" in cfg:
        cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
        cfg.pop("gamma_value", None)  # Remove "gamma_value"

    clf = svm.SVC(**cfg, random_state=42)

    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1 - np.mean(scores)  # Minimize!

def build_svm_space(rng):
    global iris
    iris = datasets.load_iris()
        # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace(seed=rng.randint(MAXINT))

    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    kernel = CategoricalHyperparameter("kernel",
                                       ["linear", "rbf", "poly", "sigmoid"],
                                       default_value="poly")
    cs.add_hyperparameter(kernel)

    # There are some hyperparameters shared by all kernels
    C = UniformFloatHyperparameter("C",
                                   0.001,
                                   1000.0,
                                   default_value=1.0,
                                   log=True)
    shrinking = CategoricalHyperparameter("shrinking", [True, False],
                                          default_value=True)
    cs.add_hyperparameters([C, shrinking])

    # Others are kernel-specific, so we can add conditions to limit the searchspace
    degree = UniformIntegerHyperparameter(
        "degree", 1, 5, default_value=3)  # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0,
                                       default_value=0.0)  # poly, sigmoid
    cs.add_hyperparameters([degree, coef0])

    use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = InCondition(child=coef0,
                            parent=kernel,
                            values=["poly", "sigmoid"])
    cs.add_conditions([use_degree, use_coef0])

    # This also works for parameters that are a mix of categorical and values
    # from a range of numbers
    # For example, gamma can be either "auto" or a fixed float
    gamma = CategoricalHyperparameter(
        "gamma", ["auto", "value"],
        default_value="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value",
                                             0.0001,
                                             8,
                                             default_value=1,
                                             log=True)
    cs.add_hyperparameters([gamma, gamma_value])
    # We only activate gamma_value if gamma is set to "value"
    cs.add_condition(
        InCondition(child=gamma_value, parent=gamma, values=["value"]))
    # And again we can restrict the use of gamma in general to the choice of the kernel
    cs.add_condition(
        InCondition(child=gamma,
                    parent=kernel,
                    values=["rbf", "poly", "sigmoid"]))
    return cs