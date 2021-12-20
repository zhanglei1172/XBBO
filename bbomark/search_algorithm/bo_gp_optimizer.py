import typing
import numpy as np
import tqdm, random
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         CategoricalHyperparameter,
                                         OrdinalHyperparameter)
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process import GaussianProcessRegressor

from bbomark.core import AbstractOptimizer
from bbomark.configspace.space import Configurations
from bbomark.core import trials
from bbomark.core.stochastic import Category, Uniform
from bbomark.core.trials import Trials
from bbomark.initial_design.sobol import SobolDesign
from bbomark.surrogate.gaussian_process import GPR_sklearn
from bbomark.acquisition_function.ei import EI_

class BOGP(AbstractOptimizer):
    def __init__(
            self,
            config_spaces,
            rng=np.random.RandomState(0),
            #  min_sample=1,
            total_limit=10,
            predict_x_best: bool = True):
        '''
        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        '''    
        AbstractOptimizer.__init__(self, config_spaces)
        # self.min_sample = min_sample
        configs = self.space.get_hyperparameters()
        self.rng = rng
        self.predict_x_best = predict_x_best
        self.dense_dimension = self.space.get_dimensions(sparse=False)

        self.initial_design = SobolDesign(self.dense_dimension,rng,
                                          ta_run_limit=total_limit)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(configs)
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        self.trials = Trials()
        self.surrogate_model = GPR_sklearn(self.dense_dimension,min_sample=self.init_budget, rng=self.rng)
        self.acquisition_func = EI_()

    def suggest(self, n_suggestions=1):
        # 只suggest 一个
        if (self.trials.trials_num) < self.init_budget:
            assert self.trials.trials_num / n_suggestions == 0
            return self.initial_design_configs[
                int(n_suggestions *
                    self.trials.trials_num):int(n_suggestions *
                                                (self.trials.trials_num + 1))]
        else:
            self.surrogate_model._train(np.array(self.trials.history), np.array(self.trials.history_y))
            X = np.atleast_2d(self.trials.history)
            sas = []
            for n in range(n_suggestions):
                suggest_array = []
                for i in range(self.hp_num):
                    best_val = self._get_x_best(self.predict_x_best, X)
                    self.acquisition_func.update(self.surrogate_model, best_val)
                    arr = self.acquisition_func.argmax(np.random.randn(1000, self.dense_dimension))
                    suggest_array.append(arr)
                    
                    

                sas.append(suggest_array)
        x = [
            Configurations.array_to_dictUnwarped(self.space, np.array(sa))
            for sa in sas
        ]
        self.trials.params_history.extend(x)
        return x, sas

    def observe(self, x, y):
        self.trials.history.extend(x)
        self.trials.history_y.extend(y)
        self.trials.trials_num += 1



    def _get_x_best(self, predict: bool,
                    X: np.ndarray) -> typing.Tuple[float, np.ndarray]:
        """Get value, configuration, and array representation of the "best" configuration.

        The definition of best varies depending on the argument ``predict``. If set to ``True``,
        this function will return the stats of the best configuration as predicted by the model,
        otherwise it will return the stats for the best observed configuration.

        Parameters
        ----------
        predict : bool
            Whether to use the predicted or observed best.

        Returns
        -------
        float
        np.ndarry
        Configuration
        """
        if predict:
            costs = list(
                map(
                    lambda x: (
                        self.gp.predict(x.reshape((1, -1)))[0][0][0],
                        x,
                    ),
                    X,
                ))
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            best_idx = np.argmin(self.trials.history_y)
            x_best_array = self.trials.history[best_idx]
            best_observation = self.trials.history_y[best_idx]

        return x_best_array, best_observation


opt_class = BOGP