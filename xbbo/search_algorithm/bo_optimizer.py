import logging
import typing
import numpy as np

from xbbo.acquisition_function.acq_optimizer import InterleavedLocalAndRandomSearch, LocalSearch, RandomScipyOptimizer, RandomSearch, ScipyGlobalOptimizer, ScipyOptimizer

from xbbo.search_algorithm.base import AbstractOptimizer
# from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace

# from xbbo.core import trials
# from xbbo.core.stochastic import Category, Uniform
from . import alg_register
from xbbo.core.trials import Trial, Trials
from xbbo.initial_design import ALL_avaliable_design
from xbbo.surrogate.gaussian_process import GPR_sklearn
from xbbo.acquisition_function.acq_func import EI_AcqFunc
from xbbo.surrogate.prf import RandomForestWithInstances
from xbbo.surrogate.sk_prf import skRandomForestWithInstances
from xbbo.surrogate.skrf import RandomForestSurrogate
from xbbo.utils.util import get_types

logger = logging.getLogger(__name__)


@alg_register.register('basic-bo')
class BO(AbstractOptimizer):
    def __init__(
            self,
            space,
            seed: int = 42,
            surrogate: str = 'gp',
            acq_func: str = 'ei',
            acq_opt: str = 'rs_ls',
            initial_design: str = 'sobol',
            #  min_sample=1,
            suggest_limit: int = np.inf,
            predict_x_best: bool = True,
            **kwargs):
        '''
        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        '''
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)
        # self.min_sample = min_sample
        # configs = self.space.get_hyperparameters()
        self.predict_x_best = predict_x_best
        self.dimension = self.space.get_dimensions()

        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )

        self.trials = Trials(self.dimension)
        if surrogate == 'gp':
            self.surrogate_model = GPR_sklearn(self.space, rng=self.rng)
        elif surrogate == 'prf':
            self.surrogate_model = RandomForestWithInstances(self.space,
                                                             rng=self.rng)
        elif surrogate == 'rf':
            self.surrogate_model = RandomForestSurrogate(self.space,
                                                         rng=self.rng)
        elif surrogate == 'sk_prf':
            self.surrogate_model = skRandomForestWithInstances(self.space,
                                                               rng=self.rng)
        else:
            raise ValueError('surrogate {} not in {}'.format(
                surrogate, ['gp', 'rf', 'prf', 'sk_prf']))

        if acq_func == 'ei':
            self.acquisition_func = EI_AcqFunc(self.surrogate_model, self.rng)
        # elif acq_func == 'rf':
        #     self.acquisition_func = None
        else:
            raise ValueError('acq_func {} not in {}'.format(acq_func, ['ei']))
        if acq_opt == 'ls':
            self.acq_maximizer = LocalSearch(self.acquisition_func, self.space,
                                             self.rng)
        elif acq_opt == 'rs':
            self.acq_maximizer = RandomSearch(self.acquisition_func,
                                              self.space, self.rng)
        elif acq_opt == 'rs_ls':
            self.acq_maximizer = InterleavedLocalAndRandomSearch(
                self.acquisition_func, self.space, self.rng)
        elif acq_opt == 'scipy':
            self.acq_maximizer = ScipyOptimizer(self.acquisition_func,
                                                self.space, self.rng)
        elif acq_opt == 'scipy_global':
            self.acq_maximizer = ScipyGlobalOptimizer(self.acquisition_func,
                                                      self.space, self.rng)
        elif acq_opt == 'r_scipy':
            self.acq_maximizer = RandomScipyOptimizer(self.acquisition_func,
                                                      self.space, self.rng)
        else:
            raise ValueError('acq_opt {} not in {}'.format(
                acq_opt,
                ['ls', 'rs', 'rs_ls', 'scipy', 'scipy_global', 'r_scipy']))
        logger.info(
            "Execute Bayesian optimization...\n [Using ({})surrogate, ({})acquisition function, ({})acquisition optmizer]"
            .format(surrogate, acq_func, acq_opt))

    def _suggest(self, n_suggestions=1):
        trial_list = []
        # currently only suggest one
        if (self.trials.trials_num) < self.init_budget:
            assert self.trials.trials_num % n_suggestions == 0
            configs = self.initial_design_configs[
                int(n_suggestions *
                    self.trials.trials_num):int(n_suggestions *
                                                (self.trials.trials_num + 1))]
            for config in configs:
                trial_list.append(
                    Trial(configuration=config,
                          config_dict=config.get_dictionary(),
                          array=config.get_array(sparse=False)))
        else:
            self.surrogate_model.train(
                np.asarray(self.trials.get_array()),
                np.asarray(self.trials.get_history()[0]))
            configs = []
            _, best_val = self._get_x_best(self.predict_x_best)
            self.acquisition_func.update(surrogate_model=self.surrogate_model,
                                         y_best=best_val)
            configs = self.acq_maximizer.maximize(self.trials,
                                                  1000,
                                                  drop_self_duplicate=True)
            _idx = 0
            for n in range(n_suggestions):
                while _idx < len(configs):  # remove history suggest
                    if not self.trials.is_contain(configs[_idx]):
                        config = configs[_idx]
                        configs.append(config)
                        trial_list.append(
                            Trial(configuration=config,
                                  config_dict=config.get_dictionary(),
                                  array=config.get_array(sparse=False)))
                        _idx += 1

                        break
                    _idx += 1
                else:
                    assert False, "no more configs can be suggest"

        return trial_list

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial)

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
            X = self.trials.get_array()
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
            x_best_array = self.trials.get_array()[best_idx]
            best_observation = self.trials.best_observe_value

        return x_best_array, best_observation


opt_class = BO