import numpy as np
import time, yaml
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
    def __init__(self, dim=10, rng=np.random.RandomState(), **kwargs):
        self.dim      = dim
        self.keys = ["x_{}".format(i) for i in range(self.dim)]
        super().__init__(rng)
        self.get_configuration_space()
        

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
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
    def __init__(self, dim=2, rng=np.random.RandomState(), **kwargs) -> None:
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
        if hasattr(self, "configuration_space"):
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
    def __init__(self, dim:int=2, rng=np.random.RandomState(), **kwargs) -> None:
        self.dim = dim
        self.keys = ["x_{}".format(i) for i in range(self.dim)]
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = []
        for key in self.keys:
            x.append(config[key])
        x = np.array(x)
        y = sum(100*(x[:-1]**2-x[1:])**2 + (x[:-1]-1)**2)

        return {Key.FUNC_VALUE: y}
    


    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for key in self.keys:
            self.configuration_space.add_hyperparameter(UniformFloatHyperparameter(key, -5, 10, default_value=-3))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Rosenbrock'}

@problem_register.register('Forrester')
class Forrester(AbstractBenchmark):
    def __init__(self, dim:int=1, rng=np.random.RandomState(), **kwargs) -> None:
        assert dim == 1, "ERROR: Current only support 1dim Forrester"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()


    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = config["x"]
        y = (6.*x - 2.)**2 * np.sin(12.*x-4.)

        return {Key.FUNC_VALUE: y}
        
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = CS.ConfigurationSpace()
        self.configuration_space.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=0., upper=1.))
        return self.configuration_space

    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Forrester'}

@problem_register.register('Sinusoid')
class Sinusoid(AbstractBenchmark):
    def __init__(self, dim:int=1, rng=np.random.RandomState(), **kwargs) -> None:
        assert dim == 1, "ERROR: Current only support 1dim Sinusoid"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()


    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = config["x"]
        y = np.sin(3.0*x) + x**2 - 0.7*x

        return {Key.FUNC_VALUE: y}
        
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = CS.ConfigurationSpace()
        self.configuration_space.add_hyperparameter(
            CS.UniformFloatHyperparameter("x", lower=-1., upper=2.))
        return self.configuration_space

    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Sinusoid'}

@problem_register.register('StyblinskiTang')
class StyblinskiTang(AbstractBenchmark):
    def __init__(self, dim:int=2, rng=np.random.RandomState(), **kwargs) -> None:
        self.dim = dim
        self.keys = ["x_{}".format(i) for i in range(self.dim)]
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = []
        for key in self.keys:
            x.append(config[key])
        x = np.array(x)
        y = .5 * np.sum(x**4 - 16 * x**2 + 5*x)

        return {Key.FUNC_VALUE: y}

    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for key in self.keys:
            self.configuration_space.add_hyperparameter(UniformFloatHyperparameter(key, -5, 5))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: StyblinskiTang'}
    
    def get_minimum(self):
        return -39.16599 * self.dim

@problem_register.register('Michalewicz')
class Michalewicz(AbstractBenchmark):
    def __init__(self, dim:int=2, rng=np.random.RandomState(), m=10) -> None:
        self.dim = dim
        self.m = m
        self.keys = ["x_{}".format(i) for i in range(self.dim)]
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = []
        for key in self.keys:
            x.append(config[key])
        x = np.array(x)
        N = len(x)
        n = np.arange(N) + 1

        a = np.sin(x)
        b = np.sin(n * x**2 / np.pi)
        b **= 2*self.m
        y = - np.sum(a * b)

        return {Key.FUNC_VALUE: y}

    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for key in self.keys:
            self.configuration_space.add_hyperparameter(UniformFloatHyperparameter(key, 0., np.pi))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Michalewicz'}
    
    def get_minimum(self):
        minimums = {
            2: -1.8013,
            5: -4.687658,
            10: -9.66015
        }
        assert self.dim in minimums, \
            f"global minimum for dimensions={self.dim} not known"
        return minimums[self.dim]

@problem_register.register('Hartmann')
class Hartmann(AbstractBenchmark):
    def __init__(self, dim, A, P, alpha=np.array([1.0, 1.2, 3.0, 3.2]), rng=np.random.RandomState(), **kwargs) -> None:
        self.dim = dim
        self.keys = ["x_{}".format(i) for i in range(self.dim)]
        self.A = A
        self.P = P
        self.alpha = alpha
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x = []
        for key in self.keys:
            x.append(config[key])
        x = np.array(x)
        r = np.sum(self.A * np.square(x - self.P))
        y = - np.dot(np.exp(-r), self.alpha)

        return {Key.FUNC_VALUE: y}

    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for key in self.keys:
            self.configuration_space.add_hyperparameter(UniformFloatHyperparameter(key, 0, 1))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Hartmann'}
    
@problem_register.register('Hartmann3D')
class Hartmann3D(Hartmann):
    def __init__(self, rng=np.random.RandomState(), **kwargs) -> None:
        dim = 3
        self.keys = ["x_{}".format(i) for i in range(self.dim)]
        A = np.array([[3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0],
                      [3.0, 10.0, 30.0],
                      [0.1, 10.0, 35.0]])
        P = 1e-4 * np.array([[3689, 1170, 2673],
                             [4699, 4387, 7470],
                             [1091, 8732, 5547],
                             [381,  5743, 8828]])
        super().__init__(dim, A, P, rng=rng)
        self.get_configuration_space()

    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Hartmann3D'}


    def get_minimum(self):
        return -3.86278

@problem_register.register('Hartmann6D')
class Hartmann6D(Hartmann):

    def __init__(self, rng=np.random.RandomState(), **kwargs) -> None:
        A = np.array([[10.0,  3.0, 17.0,  3.5,  1.7,  8.0],
                      [0.05, 10.0, 17.0,  0.1,  8.0, 14.0],
                      [3.0,  3.5,  1.7, 10.0, 17.0,  8.0],
                      [17.0,  8.0,  0.05, 10.0,  0.1, 14.0]])
        P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                             [2329, 4135, 8307, 3736, 1004, 9991],
                             [2348, 1451, 3522, 2883, 3047, 6650],
                             [4047, 8828, 8732, 5743, 1091,  381]])
        super(Hartmann6D, self).__init__(dim=6, A=A, P=P, rng=rng)

    def get_minimum(self):
        return -3.32237

@problem_register.register('GoldsteinPrice')
class GoldsteinPrice(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(), **kwargs) -> None:
        assert dim == 2, "ERROR: Current only support 2dim GoldsteinPrice"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x, y = config['x'], config['y']
        a = 1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y + 3*y**2)
        b = 30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2)

        return {Key.FUNC_VALUE: a*b}
    


    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x1 = UniformFloatHyperparameter("x", -2, 2)
        x2 = UniformFloatHyperparameter("y", -2, 2)
        self.configuration_space.add_hyperparameters([x1, x2])
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: GoldsteinPrice'}

    def get_minimum(self):
        return 3.

@problem_register.register('SixHumpCamel')
class SixHumpCamel(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(), **kwargs) -> None:
        assert dim == 2, "ERROR: Current only support 2dim SixHumpCamel"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        x, y = config['x'], config['y']
        r = (4 - 2.1 * x**2 + x**4/3) * x**2 + x*y + (-4 + 4 * y**2) * y**2

        return {Key.FUNC_VALUE: r}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x1 = UniformFloatHyperparameter("x", -3., 3.)
        x2 = UniformFloatHyperparameter("y", -2., 2.)
        self.configuration_space.add_hyperparameters([x1, x2])
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: SixHumpCamel'}

    def get_minimum(self):
        return -1.0316

@problem_register.register('Bliznyuk')
class Bliznyuk(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(), **kwargs) -> None:
        assert dim == 2, "ERROR: Current only support 4dim Bliznyuk"
        self.dim = dim
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, **kwargs):
        m, d, l, tau = config['m'], config['d'], config['l'], config['tau']
        def cost(s_, t_, m_, d_, l_, tau_):
            first_term = m_ / np.sqrt(4 * np.pi * d_ * t_) * np.exp(-(s_ ** 2) / (4 * d_ * t_))
            second_term = np.where(t_ - tau_ > 0, m_ / np.sqrt(4 * np.pi * d_ * (t_ - tau_)) * np.exp(-((s_ - l_) ** 2) / (4 * d_ * (t_ - tau_))), 0.0)
            return first_term + second_term
        
        tot = 0.0
        for s in [0, 1, 2.5]:
            for t in [15, 30, 45, 60]:
                tot += (cost(s, t, m, d, l, tau) - cost(s, t, 10, 0.07, 1.505, 30.1525)) ** 2
        return {Key.FUNC_VALUE: tot}

    @AbstractBenchmark._check_configuration
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x1 = UniformFloatHyperparameter("m", 7., 13.)
        x2 = UniformFloatHyperparameter("d", 0.02, 0.12)
        x3 = UniformFloatHyperparameter("l", 0.01, 3)
        x4 = UniformFloatHyperparameter("tau", 30.01, 30.295)
        self.configuration_space.add_hyperparameters([x1, x2,x3,x4])
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Bliznyuk'}

    def get_minimum(self):
        return 0


@problem_register.register('ZDT1')
class ZDT1(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(), **kwargs) -> None:
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
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        x0 = UniformFloatHyperparameter("x1", 0, 1)
        x1 = UniformFloatHyperparameter("x2", 0, 1)
        self.configuration_space.add_hyperparameters([x0, x1])
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: ZTD1', 'desc': '2 obj'}

@problem_register.register('ContingOnes')
class CountingOnes(AbstractBenchmark):
    def __init__(self, n_categorical=1, n_continuous=1, rng=np.random.RandomState(), **kwargs) -> None:
        self.n_categorical = n_categorical
        self.n_continuous = n_continuous
        self.float_keys = ["float_{}".format(i) for i in range(self.n_continuous)]
        self.cat_keys = ["cat_{}".format(i) for i in range(self.n_categorical)]
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=100, **kwargs):
        y = 0
        for key in self.float_keys:
            samples = self.rng.binomial(1, config[key], int(budget))
            y += np.mean(samples)
        for key in self.cat_keys:
            y += config[key]

        return {Key.FUNC_VALUE: -y, Key.BUDGET:budget}
    


    def objective_function_test(self, config, **kwargs):
        return {Key.FUNC_VALUE: -np.sum(config.get_array())}
    
    def get_configuration_space(self,):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = ConfigurationSpace(seed=self.rng.randint(MAXINT))
        for key in self.float_keys:
            self.configuration_space.add_hyperparameter(CS.CategoricalHyperparameter(key, [0, 1]))
        for key in self.cat_keys:
            self.configuration_space.add_hyperparameter(CS.UniformFloatHyperparameter(key, lower=0, upper=1))
        return self.configuration_space
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: Couning Ones(Multi-fidelity)'}

@problem_register.register('SVM')
class SVM_hyperparam_search(AbstractBenchmark):
    def __init__(self, dim=2, rng=np.random.RandomState(), **kwargs) -> None:
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
        if hasattr(self, "configuration_space"):
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
    
@problem_register.register('NasBench201')
class NasBench201(AbstractBenchmark):
    def __init__(self, dataset_name='cifar10-valid', input_dir='./nasbench201/',rng=np.random.RandomState()):
        self.INPUT = 'input'
        self.OUTPUT = 'output'
        self.OPS = ['avg_pool_3x3', 'nor_conv_1x1', 'nor_conv_3x3', 'none', 'skip_connect']
        self.NUM_OPS = len(self.OPS)
        self.OP_SPOTS = 6
        self.dataset = dataset_name
        from nas_201_api import NASBench201API as API
        import os
        self.nasbench = API(os.path.join(input_dir, 'NAS-Bench-201-v1_1-096897.pth'))
        super().__init__(rng)
        self.get_configuration_space()

    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=None, deterministic=False,**kwargs):
          # True if used for getting the minimum mean validation error
        ops = []
        for i in range(self.OP_SPOTS):
            ops.append(config[f"op_{i}"])
        string = self._get_string_from_ops(ops)
        index = self.nasbench.query_index_by_arch(string)
        results = self.nasbench.query_by_index(index, self.dataset)
        accs = []
        times = []
        for key in results.keys():
            accs.append(results[key].get_eval('x-valid')['accuracy'])
            times.append(results[key].get_eval('x-valid')['all_time'])
        if not deterministic:
            sample_id = self.rng.choice(len(accs))
            loss = round(100 - accs[sample_id], 10) / 100.
            time = times[sample_id]
            return {Key.FUNC_VALUE: loss, Key.COST: time}
        else:
            loss = round(100 - np.mean(accs), 10) / 100.
            time = np.mean(times)
            return {Key.FUNC_VALUE: loss, Key.COST: time}

    def _get_string_from_ops(self, ops):
        # given a list of operations, get the string
        strings = ['|']
        nodes = [0, 0, 1, 0, 1, 2]
        for i, op in enumerate(ops):
            strings.append(op + '~{}|'.format(nodes[i]))
            if i < len(nodes) - 1 and nodes[i + 1] == 0:
                strings.append('+|')
        return ''.join(strings)

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = CS.ConfigurationSpace()
        for i in range(self.OP_SPOTS):
            self.configuration_space.add_hyperparameter(CS.CategoricalHyperparameter(f"op_{i}", self.OPS))
        return self.configuration_space

    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, deterministic=True, **kwargs)

    def get_minimum(self):
        if self.dataset == 'cifar10-valid':
            return 0.151080000098
        elif self.dataset == 'cifar100':
            return 0.3868
        elif self.dataset == 'ImageNet16-120':
            return 0.61333333374
        else:
            raise NotImplementedError

    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: NasBench201'}

@problem_register.register('FCNet')
class FCNet(AbstractBenchmark):
    '''
    wget -P ./datasets/ http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz
    tar xf fcnet_tabular_benchmarks.tar.gz
    install HPOBench: https://github.com/automl/nas_benchmarks
    '''
    def __init__(self, dataset_name="protein", input_dir='./datasets', rng=np.random.RandomState(), **kwargs):
        from pathlib import Path
        from tabular_benchmarks import (FCNetProteinStructureBenchmark,
                                        FCNetSliceLocalizationBenchmark,
                                        FCNetNavalPropulsionBenchmark,
                                        FCNetParkinsonsTelemonitoringBenchmark)
        data_dir = Path(input_dir).joinpath("fcnet_tabular_benchmarks")
        def _get_benchmark(dataset_name, data_dir):
            benchmarks = dict(
                protein=FCNetProteinStructureBenchmark(data_dir=data_dir),
                slice=FCNetSliceLocalizationBenchmark(data_dir=data_dir),
                naval=FCNetNavalPropulsionBenchmark(data_dir=data_dir),
                parkinsons=FCNetParkinsonsTelemonitoringBenchmark(data_dir=data_dir))
            return benchmarks.get(dataset_name)
        benchmark = _get_benchmark(dataset_name, data_dir)
        if benchmark is None:
            raise ValueError("dataset name not recognized!")
        self.benchmark = benchmark
        self.dataset_name = dataset_name
        self.data_dir = data_dir
        super().__init__(rng)
        self.get_configuration_space()
    
    @AbstractBenchmark._check_configuration
    def objective_function(self, config, budget=None, deterministic=False,**kwargs):
        y, cost = self.benchmark.objective_function(config)
        return {Key.FUNC_VALUE: y, Key.COST: cost}

    def get_configuration_space(self):
        if hasattr(self, "configuration_space"):
            return self.configuration_space
        self.configuration_space = self.benchmark.get_configuration_space()
        return self.configuration_space

    def get_minimum(self):

        base_path = self.data_dir.joinpath(self.dataset_name)
        base_path.mkdir(parents=True, exist_ok=True)

        path = base_path.joinpath("minimum.yaml")

        if path.exists():
            with path.open('r') as f:
                val_error_min = yaml.safe_load(f).get("val_error_min")
        else:
            config_dict, val_error_min, \
                test_error_min = self.benchmark.get_best_configuration()
            d = dict(config_dict=config_dict,
                     val_error_min=float(val_error_min),
                     test_error_min=float(test_error_min))
            with path.open('w') as f:
                yaml.dump(d, f)

        return float(val_error_min)
    
    def objective_function_test(self, config, **kwargs):
        return self.objective_function(config, **kwargs)
    
    @staticmethod
    def get_meta_information():
        return {'name': 'Test Function: FCNet'}

