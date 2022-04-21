import logging, typing
import numpy as np
from xbbo.acquisition_function.transfer.mogp import MoGP_AcqFunc
from xbbo.acquisition_function.transfer.taf import TAF_AcqFunc
from xbbo.acquisition_function.acq_func import EI_AcqFunc
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.surrogate.gaussian_process import GPR_sklearn
from . import alg_register
from xbbo.core.trials import Trial, Trials
from xbbo.initial_design import ALL_avaliable_design
from xbbo.acquisition_function.acq_optimizer import InterleavedLocalAndRandomSearch, LocalSearch, RandomScipyOptimizer, RandomSearch, ScipyGlobalOptimizer, ScipyOptimizer
from xbbo.surrogate.transfer.weight_stategy import KernelRegress, RankingWeight, ZeroWeight
from xbbo.surrogate.transfer.tst import BaseModel, TST_surrogate

logger = logging.getLogger(__name__)


@alg_register.register('bo-transfer')
class SMBO(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 initial_design: str = 'sobol',
                 suggest_limit: int = np.inf,
                 surrogate: str = 'gp',
                 acq_func: str = 'ei',
                 acq_opt: str = 'rs_ls',
                 predict_x_best: bool = False,
                 weight_srategy: str = 'kernel',
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)
        self.predict_x_best = predict_x_best
        self.dimension = self.space.get_dimensions()

        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        self.trials = Trials(dim=self.dimension)

        # self.rho = kwargs.get("rho", 1)
        self.bandwidth = kwargs.get("bandwdth", 0.1)
        self.base_models = kwargs.get("base_models")
        if self.base_models:
            assert isinstance(self.base_models[0], BaseModel)
            if surrogate == 'gp':
                self.surrogate_model = GPR_sklearn(self.space, rng=self.rng)
            elif surrogate == 'tst':
                self.surrogate_model = TST_surrogate(self.space,
                                                     self.base_models,
                                                     rng=self.rng)
        else:
            raise NotImplementedError()
        if weight_srategy == 'kernel':
            self.weight_sratety = KernelRegress(self.space, self.base_models,
                                                self.surrogate_model, self.rng)
        elif weight_srategy == 'rw':
            self.weight_sratety = RankingWeight(self.space,
                                                self.base_models,
                                                self.surrogate_model,
                                                self.rng,
                                                budget=suggest_limit,
                                                is_purn=True)
        elif weight_srategy == 'zero':
            self.weight_sratety = ZeroWeight(self.space, self.base_models,
                                             self.surrogate_model, self.rng)
        else:
            raise NotImplementedError()

        if acq_func == 'mogp':
            self.acquisition_func = MoGP_AcqFunc(self.surrogate_model,
                                                 self.base_models, self.rng)
        elif acq_func == 'taf':
            self.acquisition_func = TAF_AcqFunc(self.surrogate_model,
                                                self.base_models, self.rng)
        elif acq_func == 'ei':
            self.acquisition_func = EI_AcqFunc(self.surrogate_model, self.rng)
        else:
            raise NotImplementedError()

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
            # update target surrogate model
            self.surrogate_model.train(
                np.asarray(self.trials.get_array()),
                np.asarray(self.trials.get_history()[0]))
            # calculate base incuments (only use for acq base EI)
            observed_X = self.trials.get_array()
            base_incuments = []
            for model in self.base_models:  # TODO make sure untransform ?
                base_incuments.append(model.predict(observed_X, None)[0].min())
            _, best_val = self._get_x_best(self.predict_x_best)
            self.acquisition_func.update(surrogate_model=self.surrogate_model,
                                         y_best=best_val,
                                         _base_incuments=base_incuments)
            # caculate weight for base+target model
            weight = self.weight_sratety.get_weight(self.trials)
            self.surrogate_model.update_weight(weight)
            self.acquisition_func.update_weight(weight)
            # acq maximize
            configs = []
            configs = self.acq_maximizer.maximize(self.trials,
                                                  1000,
                                                  drop_self_duplicate=True,
                                                  _sorted=True)
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
                # surrogate = TST_surrogate(self.gps, self.target_model,
                #   self.similarity, self.rho)

        return trial_list

    def _observe(self, trial_list):
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


opt_class = SMBO
