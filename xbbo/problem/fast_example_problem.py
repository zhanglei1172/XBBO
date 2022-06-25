import numpy as np
import time
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
from ConfigSpace import ConfigurationSpace
import ConfigSpace as CS
from ConfigSpace.conditions import InCondition, LessThanCondition
from ConfigSpace.hyperparameters import \
    CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter
from .base import AbstractBenchmark
from . import problem_register

from xbbo.core.constants import MAXINT, Key

@problem_register.register('Ackley')
class Ackley(AbstractBenchmark):
    def __init__(self, dim=10, rng=np.random.RandomState(42)):
        self.dims      = dim
        self.keys = ["x_{}".format(i) for i in range(self.dims)]
        super().__init__(rng)
        self.get_configuration_space()
        

    def get_configuration_space(self):
        if hasattr(self, "cs"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for k in self.keys:
            self.configuration_space.add_hyperparameter(UniformFloatHyperparameter(k, -5, 10))
        return self.configuration_space
    
    @AbstractBenchmark._check_configuration
    def objective_function(self, config_dict, info=None, **kwargs):
        # assert len(config_dict) == self.dims
        l = []
        for k in self.keys:
            l.append(config_dict[k])
        array = np.array(l)
        result = (-20*np.exp(-0.2 * np.sqrt(np.inner(array,array) / len(array) )) -np.exp(np.cos(2*np.pi*array).sum() /len(array)) + 20 +np.e )
                
        return {Key.FUNC_VALUE: result}
    
    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)

    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Ackley'}

@problem_register.register('Branin')
class Branin(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(42)) -> None:
        assert dim == 2, "ERROR: Current only support 2dim branin"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x1, x2 = config['x1'], config['x2']
        y = (x2 - 5.1 / (4 * np.pi ** 2) * x1 ** 2 + 5 / np.pi * x1 - 6) ** 2 \
            + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x1) + 10

        return {Key.FUNC_VALUE: y}
    


    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "cs"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x1 = UniformFloatHyperparameter("x1", -5, 10, default_value=0)
        x2 = UniformFloatHyperparameter("x2", 0, 15, default_value=0)
        self.configuration_space.add_hyperparameters([x1, x2])
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Branin'}

@problem_register.register('Rosenbrock')
class Rosenbrock(AbstractBenchmark):
    def __init__(self, dim:int=2, rng=np.random.RandomState(42)) -> None:
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = []
        for h in config:
            x.append(config[h])
        x = np.array(x)
        y = sum(100*(x[:-1]**2-x[1:])**2 + (x[:-1]-1)**2)

        return {Key.FUNC_VALUE: y}
    


    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "cs"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for i in range(self.dim):
            self.configuration_space.add_hyperparameter(UniformFloatHyperparameter("x{}".format(i), -5, 10, default_value=-3))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Rosenbrock'}

@problem_register.register('ZDT1')
class ZDT1(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(42)) -> None:
        assert dim == 2, "ERROR: Current only support 2dim ZDT1"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        n = 30
        x1, x2 = config['x1'], config['x2']
        sigma = x2 # sum(input_x[1:])
        g = 1 + sigma * 9 / (n - 1)
        h = 1 - (x1 / g)**0.5

        return {Key.FUNC_VALUE: (x1, g*h)}
    


    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "cs"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x0 = UniformFloatHyperparameter("x1", 0, 1)
        x1 = UniformFloatHyperparameter("x2", 0, 1)
        self.configuration_space.add_hyperparameters([x0, x1])
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Branin'}

@problem_register.register('ContingOnes')
class CountingOnes(AbstractBenchmark):
    def __init__(self, n_categorical=1, n_continuous=1, rng=np.random.RandomState(42)) -> None:
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=100, **kwargs):
        y = 0
        for h in config:
            if 'float' in h:
                samples = self.rng.binomial(1, config[h], int(budget))
                y += np.mean(samples)
            else:
                y += config[h]

        return {Key.FUNC_VALUE: -y, Key.BUDGET:budget}
    


    def objective_function_test(self, config, **kwargs):
        return {Key.FUNC_VALUE: -np.sum(config.get_array())}
    
    def get_configuration_space(self,):
        if hasattr(self, "cs"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for i in range(self.n_categorical):
            self.configuration_space.add_hyperparameter(CS.CategoricalHyperparameter("cat_%d" % i, [0, 1]))
        for i in range(self.n_continuous):
            self.configuration_space.add_hyperparameter(CS.UniformFloatHyperparameter('float_%d' % i, lower=0, upper=1))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Couning Ones(Multi-fidelity)'}

@problem_register.register('SVM')
class SVM_hyperparam_search(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(42)) -> None:
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, cfg, **kwargs):
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        # And for gamma, we set it to a fixed value or to "auto" (if used)
        if "gamma" in cfg:
            cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
            cfg.pop("gamma_value", None)  # Remove "gamma_value"

        clf = svm.SVC(**cfg, random_state=self.rng)

        scores = cross_val_score(clf, self.iris.data, self.iris.target, cv=5)

        return {Key.FUNC_VALUE: 1-np.mean(scores)}
    


    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "cs"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        self.iris = datasets.load_iris()
            # Build Configuration Space which defines all parameters and their ranges
        # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
        kernel = CategoricalHyperparameter("kernel",
                                        ["linear", "rbf", "poly", "sigmoid"],
                                        default_value="poly")
        self.configuration_space.add_hyperparameter(kernel)

        # There are some hyperparameters shared by all kernels
        C = UniformFloatHyperparameter("C",
                                    0.001,
                                    1000.0,
                                    default_value=1.0,
                                    log=True)
        shrinking = CategoricalHyperparameter("shrinking", [True, False],
                                            default_value=True)
        self.configuration_space.add_hyperparameters([C, shrinking])

        # Others are kernel-specific, so we can add conditions to limit the searchspace
        degree = UniformIntegerHyperparameter(
            "degree", 1, 5, default_value=3)  # Only used by kernel poly
        coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0,
                                        default_value=0.0)  # poly, sigmoid
        self.configuration_space.add_hyperparameters([degree, coef0])

        use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
        use_coef0 = InCondition(child=coef0,
                                parent=kernel,
                                values=["poly", "sigmoid"])
        self.configuration_space.add_conditions([use_degree, use_coef0])

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
        self.configuration_space.add_hyperparameters([gamma, gamma_value])
        # We only activate gamma_value if gamma is set to "value"
        self.configuration_space.add_condition(
            InCondition(child=gamma_value, parent=gamma, values=["value"]))
        # And again we can restrict the use of gamma in general to the choice of the kernel
        self.configuration_space.add_condition(
            InCondition(child=gamma,
                        parent=kernel,
                        values=["rbf", "poly", "sigmoid"]))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: SVM hyper parameters search'}