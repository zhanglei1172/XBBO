import typing
import numpy as np
# import tqdm, random
from ConfigSpace.hyperparameters import (UniformIntegerHyperparameter,
                                         UniformFloatHyperparameter,
                                         CategoricalHyperparameter,
                                         OrdinalHyperparameter)
from xbbo.acquisition_function.acq_optimizer import InterleavedLocalAndRandomSearch, LocalSearch
# from sklearn.gaussian_process.kernels import Kernel
# from sklearn.gaussian_process import GaussianProcessRegressor

from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration

# from xbbo.core import trials
# from xbbo.core.stochastic import Category, Uniform
from xbbo.core.trials import Trial, Trials
from xbbo.initial_design.sobol import SobolDesign
from xbbo.surrogate.gaussian_process import GPR_sklearn
from xbbo.acquisition_function.ei import EI_AcqFunc


class BOGP(AbstractOptimizer):
    def __init__(
            self,
            config_spaces: DenseConfiguration,
            seed: int = 42,
            #  min_sample=1,
            total_limit: int = 10,
            predict_x_best: bool = True):
        '''
        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        '''
        AbstractOptimizer.__init__(self, config_spaces)
        # self.min_sample = min_sample
        configs = self.space.get_hyperparameters()
        self.rng = np.random.RandomState(seed)
        self.predict_x_best = predict_x_best
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)

        self.initial_design = SobolDesign(self.space,
                                          self.rng,
                                          ta_run_limit=total_limit)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(configs)
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        self.trials = Trials(sparse_dim=self.sparse_dimension, dense_dim=self.dense_dimension)
        self.surrogate_model = GPR_sklearn(self.space, rng=self.rng)
        self.acquisition_func = EI_AcqFunc(self.surrogate_model, self.rng)
        # self.acq_maximizer = LocalSearch(self.acquisition_func, self.space, self.rng)
        self.acq_maximizer = InterleavedLocalAndRandomSearch(self.acquisition_func,self.space,self.rng)

    def suggest(self, n_suggestions=1):
        trial_list = []
        # 只suggest 一个
        if (self.trials.trials_num) < self.init_budget:
            assert self.trials.trials_num % n_suggestions == 0
            configs = self.initial_design_configs[
                int(n_suggestions *
                    self.trials.trials_num):int(n_suggestions *
                                                (self.trials.trials_num + 1))]
            for config in configs:
                trial_list.append(Trial(configuration=config,config_dict=config.get_dictionary(), sparse_array=config.get_sparse_array()))
        else:
            self.surrogate_model._train(np.asarray(self.trials.his_sparse_array),
                                        np.asarray(self.trials._his_observe_value))
            configs = []
            _, best_val = self._get_x_best(self.predict_x_best)
            self.acquisition_func.update(surrogate_model=self.surrogate_model, y_best=best_val)
            configs = self.acq_maximizer.maximize(self.trials,1000, drop_self_duplicate=True)
            _idx = 0
            for n in range(n_suggestions): 
                while _idx < len(configs): # remove history suggest
                    if not self.trials.is_contain(configs[_idx]):
                        config = configs[_idx]
                        configs.append(config)
                        trial_list.append(Trial(configuration=config,config_dict=config.get_dictionary(), sparse_array=config.get_sparse_array()))
                        _idx += 1

                        break
                    _idx += 1
                else:
                    assert False, "no more configs can be suggest"
                # config = self.acquisition_func.argmax( # TODO argmax valid sample
                #     self.space.sample_configuration(1000))
                # arr = self.acquisition_func.argmax( # TODO argmax valid sample
                #     self.rng.rand(1000, self.dense_dimension))
                # config = DenseConfiguration.from_sparse_array(self.space, arr)


        # x = [
        #     config.get_dictionary()
        #     # DenseConfiguration.sparse_array_to_dict(self.space, config.get_sparse_array())
        #     for config in configs
        # ]
        # self.trials.params_history.extend(x)
        
        return trial_list

    def observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)
        # self.trials.history.extend(x)
        # self.trials.history_y.extend(y)
        # self.trials.trials_num += 1

    def _get_x_best(self, predict: bool) -> typing.Tuple[float, np.ndarray]:
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
            X = self.trials.his_sparse_array
            costs = list(
                map(
                    lambda x: (
                        self.surrogate_model.predict(x.reshape((1, -1)))[0][0],
                        x,
                    ),
                    X,
                ))
            costs = sorted(costs, key=lambda t: t[0])
            x_best_array = costs[0][1]
            best_observation = costs[0][0]
            # won't need log(y) if EPM was already trained on log(y)
        else:
            best_idx = self.trials.best_id
            x_best_array = self.trials.his_sparse_array[best_idx]
            best_observation = self.trials.best_observe_value

        return x_best_array, best_observation


opt_class = BOGP