import glob
import logging
from posixpath import basename
import typing
import numpy as np
from matplotlib import pyplot as plt

# import tqdm, random
from xbbo.acquisition_function.acq_optimizer import InterleavedLocalAndRandomSearch, LocalSearch, RandomScipyOptimizer, RandomSearch, ScipyGlobalOptimizer, ScipyOptimizer
from xbbo.initial_design import ALL_avaliable_design

from . import alg_register
from xbbo.acquisition_function.ei import EI, EI_AcqFunc
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

        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)

        self.rho = kwargs.get("rho", 0.75)
        self.bandwidth = kwargs.get("bandwdth", 0.4)
        if surrogate == 'gp':
            base_models = kwargs.get("base_models")
            if base_models:
                assert isinstance(base_models[0], BaseModel)
            self.surrogate_model = TST_surrogate(self.space,
                                                 base_models,
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

    def prepare(self,
                old_D_x_params,
                old_D_y,
                new_D_x_param,
                sort_idx=None,
                params=True):
        if params:
            old_D_x = []
            for insts_param in old_D_x_params:
                insts_feature = []
                for inst_param in insts_param:
                    array = DenseConfiguration.dict_to_array(
                        self.space, inst_param)
                    insts_feature.append(
                        self.array_to_feature(array, self.dense_dimension))
                old_D_x.append(np.asarray(insts_feature))
            insts_feature = []
            if new_D_x_param:
                for inst_param in new_D_x_param:
                    array = DenseConfiguration.dict_to_array(
                        self.space, inst_param)
                    insts_feature.append(
                        self.array_to_feature(array, self.dense_dimension))
                new_D_x = (np.asarray(insts_feature))
                self.candidates = new_D_x
            else:
                self.candidates = None
        else:
            old_D_x = []
            for insts_param in old_D_x_params:
                # insts_feature = []
                old_D_x.append(insts_param[:, sort_idx])

            if new_D_x_param is not None:
                new_D_x = new_D_x_param[:, sort_idx]
                self.candidates = new_D_x
            else:
                self.candidates = None

        self.old_D_num = len(old_D_x)
        self.gps = []
        for d in range(self.old_D_num):
            # self.gps.append(GaussianProcessRegressor())
            # observed_idx = np.random.randint(0, len(old_D_y[d]), size=50)
            # observed_idx = np.random.randint(0, len(old_D_y[d]), size=len())
            observed_idx = list(range(len(old_D_y[d])))
            # observed_idx = np.random.choice(len(old_D_y[d]), size=50, replace=False)
            x = (old_D_x[d][observed_idx, :])  # TODO
            y = (old_D_y[d][observed_idx])
            # train_yvar = np.full_like(y, self.noise_std ** 2)
            # self.gps.append(get_fitted_model(x, y, train_yvar))
            self.gps.append(GaussianProcessRegressorARD_gpy(self.hp_num))
            self.gps[-1].fit(x, y)
        # print(1)
        # if new_D_x is not None:
        #     candidates = new_D_x
        # else:  #
        #     raise NotImplemented
        # self.candidates = candidates

    def _kendallTauCorrelation(self, base_model_means, y):
        if y is None or len(y) < 2:
            return np.full(base_model_means.shape[0], self.rho)
        rank_loss = (base_model_means[..., None] <
                     base_model_means[..., None, :]) ^ (y[..., None] <
                                                        y[..., None, :])
        t = rank_loss.mean(axis=(-1, -2)) / self.bandwidth
        return (t < 1) * (1 - t * t) * self.rho
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
        for model in self.surrogate_model.base_models:
            base_model_means.append(
                model._cached_predict(self.trials.get_sparse_array(), None)[0])
        if not base_model_means:
            return []
        base_model_means = np.stack(base_model_means)  # [model, obs_num, 1]
        return self._kendallTauCorrelation(base_model_means,
                                           np.asarray(self.trials._his_observe_value))

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
