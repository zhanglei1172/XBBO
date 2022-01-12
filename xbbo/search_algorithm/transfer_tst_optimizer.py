import logging
from posixpath import basename
import typing
import numpy as np
from matplotlib import pyplot as plt
# import tqdm, random
from xbbo.acquisition_function.acq_optimizer import InterleavedLocalAndRandomSearch, LocalSearch, RandomScipyOptimizer, RandomSearch, ScipyGlobalOptimizer, ScipyOptimizer
from xbbo.initial_design import ALL_avaliable_design

from . import alg_register
from xbbo.acquisition_function.ei import EI_AcqFunc
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace

from xbbo.core.trials import Trial, Trials
# from xbbo.surrogate.gaussian_process import GPR_sklearn, GaussianProcessRegressorARD_gpy
from xbbo.surrogate.tst import BaseModel, TST_surrogate

logger = logging.getLogger(__name__)


@alg_register.register('bo-tst')
class SMBO(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 initial_design: str = 'sobol',
                 total_limit: int = 10,
                 surrogate: str = 'gp',
                 acq_func: str = 'ei',
                 acq_opt: str = 'rs_ls',
                 predict_x_best: bool = False,
                 **kwargs):
        AbstractOptimizer.__init__(self, space, seed, **kwargs)
        self.predict_x_best = predict_x_best
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)

        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=total_limit)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)

        self.rho = kwargs.get("rho", 1)
        self.bandwidth = kwargs.get("bandwdth", 0.4)
        self.base_models = kwargs.get("base_models")
        if self.base_models:
            assert isinstance(self.base_models[0], BaseModel)
            if surrogate == 'gp':
                self.surrogate_model = TST_surrogate(self.space,
                                                     self.base_models,
                                                     self.rho,
                                                     rng=self.rng)
        else:
            raise NotImplementedError()

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

    def _kendallTauCorrelation(self, base_model_means, y):
        if y is None or len(y) < 2:
            return np.full(base_model_means.shape[0], self.rho)
        rank_loss = (base_model_means[..., None] <
                     base_model_means[..., None, :]) ^ (y[..., None] <
                                                        y[..., None, :])
        t = rank_loss.mean(axis=(-1, -2)) / self.bandwidth
        return (t < 1) * (1 - t * t) * 3 / 4
        # return self.rho * (1 - t * t) if t < 1 else 0

    def suggest(self, n_suggestions=1):
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
                          sparse_array=config.get_sparse_array()))
        else:
            self.surrogate_model.update_similarity(self._get_similarity())
            self.surrogate_model.train(
                np.asarray(self.trials.get_sparse_array()),
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
                                  sparse_array=config.get_sparse_array()))
                        _idx += 1

                        break
                    _idx += 1
                else:
                    assert False, "no more configs can be suggest"
                # surrogate = TST_surrogate(self.gps, self.target_model,
                #   self.similarity, self.rho)

        return trial_list

    def _get_similarity(self, ):
        base_model_means = []
        for model in self.base_models:
            base_model_means.append(
                model._predict_normalize(self.trials.get_sparse_array(), None)[0])
        if not base_model_means:
            return []
        base_model_means = np.stack(base_model_means)  # [model, obs_num, 1]
        return self._kendallTauCorrelation(
            base_model_means, np.asarray(self.trials._his_observe_value))

    def observe(self, trial_list):
        # print(y)
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
            X = self.trials.get_sparse_array()
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
            x_best_array = self.trials.get_sparse_array()[best_idx]
            best_observation = self.trials.best_observe_value

        return x_best_array, best_observation
