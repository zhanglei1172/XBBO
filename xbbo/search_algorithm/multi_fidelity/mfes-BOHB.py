'''
Reference: https://github.com/automl/DEHB
'''

import logging
from typing import List
import typing
from sklearn.model_selection import KFold
import numpy as np
from xbbo.acquisition_function.acq_func import EI_AcqFunc
from xbbo.acquisition_function.acq_optimizer import InterleavedLocalAndRandomSearch, LocalSearch, RandomScipyOptimizer, RandomSearch, ScipyGlobalOptimizer, ScipyOptimizer
from xbbo.initial_design import ALL_avaliable_design
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.search_algorithm.multi_fidelity.BOHB import BOHB

from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import BasicConfigGenerator
from xbbo.surrogate.transfer.rf_ensemble import RandomForestEnsemble
from xbbo.surrogate.transfer.rf_with_instances import RandomForestWithInstances
from xbbo.utils.constants import MAXINT, Key
from xbbo.utils.util import get_types
from .. import alg_register
logger = logging.getLogger(__name__)

def std_normalization(x):
    _mean = np.mean(x)
    _std = np.std(x)
    if _std == 0:
        return np.array([0.] * len(x))
    return (np.array(x) - _mean) / _std

class SMBO(AbstractOptimizer):
    def __init__(self,
                space: DenseConfigurationSpace,
                all_budgets: List,
                seed: int = 42,
                initial_design: str = 'random',
                suggest_limit: int = np.inf,
                surrogate: str = 'rf',
                # acq_func: str = 'ei',
                acq_opt: str = 'rs_ls',
                # predict_x_best: bool = False,
                weight_srategy: str = 'rank_loss_p_norm',
                fusion_method='idp',
                init_weight=None,
                update_enable=True,
                random_fraction=1 / 3,
                **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   seed=seed,
                                   suggest_limit=suggest_limit,
                                   **kwargs)
        self.logger = logger
        self.all_budgets = all_budgets
        self.update_enable = update_enable
        self.fusion_method = fusion_method
        self.dimension = self.space.get_dimensions()
        self.random_fraction = random_fraction
        self.initial_design = ALL_avaliable_design[initial_design](
            self.space, self.rng, ta_run_limit=suggest_limit, **kwargs)
        self.init_budget = self.initial_design.init_budget
        self.hp_num = len(self.space)
        self.initial_design_configs = self.initial_design.select_configurations(
        )
        if init_weight is None:
            k = len(all_budgets)
            init_weight = [1. / (k-1)] * (k-1) + [0.]
        self.logger.info("Initialize weight to %s" %
                         init_weight)    
        self.trials = Trials(dim=self.dimension)
        if surrogate == 'rf':
            self.weighted_surrogate = RandomForestEnsemble(space, all_budgets, init_weight, fusion_method,**kwargs
        )
        self.weight_srategy = weight_srategy
        if weight_srategy == 'rank_loss_p_norm':
                self.power_num = kwargs.get("power_num", 3)
        self.acquisition_func = EI_AcqFunc(self.weighted_surrogate, self.rng)    
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
        self.weight_changed_cnt = 0
        self.target_x = dict()
        self.target_y = dict()
        for budget in self.all_budgets:
            self.target_x[budget] = []
            self.target_y[budget] = []

    @staticmethod
    def calculate_preserving_order_num(y_pred, y_true):
        array_size = len(y_pred)
        assert len(y_true) == array_size

        total_pair_num, order_preserving_num = 0, 0
        for idx in range(array_size):
            for inner_idx in range(idx + 1, array_size):
                if bool(y_true[idx] > y_true[inner_idx]) == bool(y_pred[idx] > y_pred[inner_idx]):
                    order_preserving_num += 1
                total_pair_num += 1
        return order_preserving_num, total_pair_num

    def update_weight(self):

        max_budget = self.all_budgets[-1]
        incumbent_configs = self.target_x[max_budget]
        if not self.target_x[max_budget]:
            return 
        test_x = np.array(incumbent_configs)
        test_y = np.array(self.target_y[max_budget], dtype=np.float64)

        # budget_list = self.weighted_surrogate.surrogate_budget
        budget_list = self.all_budgets
        K = len(budget_list)

        if len(test_y) >= 3:
            # Get previous weights
            if self.weight_srategy == 'rank_loss_p_norm':
                preserving_order_p = list()
                preserving_order_nums = list()
                for i, r in enumerate(budget_list):
                    fold_num = 5
                    if i != K - 1:
                        mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                        tmp_y = np.reshape(mean, -1)
                        preorder_num, pair_num = self.calculate_preserving_order_num(tmp_y, test_y)
                        preserving_order_p.append(preorder_num / pair_num)
                        preserving_order_nums.append(preorder_num)
                    else:
                        if len(test_y) < 2 * fold_num:
                            preserving_order_p.append(0)
                        else:
                            # 5-fold cross validation.
                            kfold = KFold(n_splits=fold_num)
                            cv_pred = np.array([0] * len(test_y))
                            for train_idx, valid_idx in kfold.split(test_x):
                                train_configs, train_y = test_x[train_idx], test_y[train_idx]
                                valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                                types, bounds = get_types(self.space)
                                _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                                _surrogate.train(train_configs, train_y)
                                pred, _ = _surrogate.predict(valid_configs)
                                cv_pred[valid_idx] = pred.reshape(-1)
                            preorder_num, pair_num = self.calculate_preserving_order_num(cv_pred, test_y)
                            preserving_order_p.append(preorder_num / pair_num)
                            preserving_order_nums.append(preorder_num)

                trans_order_weight = np.array(preserving_order_p)
                power_sum = np.sum(np.power(trans_order_weight, self.power_num))
                if power_sum < 1e-9:
                    old_weights = list()
                    for i, r in enumerate(budget_list):
                        _weight = self.weighted_surrogate.surrogate_weight[r]
                        old_weights.append(_weight)
                    new_weights = old_weights.copy()
                else:
                    new_weights = np.power(trans_order_weight, self.power_num) / power_sum

            elif self.weight_srategy == 'rank_loss_prob':
                # For basic surrogate i=1:K-1.
                mean_list, var_list = list(), list()
                for i, r in enumerate(budget_list[:-1]):
                    mean, var = self.weighted_surrogate.surrogate_container[r].predict(test_x)
                    mean_list.append(np.reshape(mean, -1))
                    var_list.append(np.reshape(var, -1))
                sample_num = 100
                min_probability_array = [0] * K
                for _ in range(sample_num):
                    order_preseving_nums = list()

                    # For basic surrogate i=1:K-1.
                    for idx in range(K - 1):
                        sampled_y = self.rng.normal(mean_list[idx], var_list[idx])
                        _num, _ = self.calculate_preserving_order_num(sampled_y, test_y)
                        order_preseving_nums.append(_num)

                    fold_num = 5
                    # For basic surrogate i=K. cv
                    if len(test_y) < 2 * fold_num:
                        order_preseving_nums.append(0)
                    else:
                        # 5-fold cross validation.
                        kfold = KFold(n_splits=fold_num)
                        cv_pred = np.array([0] * len(test_y))
                        for train_idx, valid_idx in kfold.split(test_x):
                            train_configs, train_y = test_x[train_idx], test_y[train_idx]
                            valid_configs, valid_y = test_x[valid_idx], test_y[valid_idx]
                            types, bounds = get_types(self.space)
                            _surrogate = RandomForestWithInstances(types=types, bounds=bounds)
                            _surrogate.train(train_configs, train_y)
                            _pred, _var = _surrogate.predict(valid_configs)
                            sampled_pred = self.rng.normal(_pred.reshape(-1), _var.reshape(-1))
                            cv_pred[valid_idx] = sampled_pred
                        _num, _ = self.calculate_preserving_order_num(cv_pred, test_y)
                        order_preseving_nums.append(_num)
                    max_id = np.argmax(order_preseving_nums)
                    min_probability_array[max_id] += 1
                new_weights = np.array(min_probability_array) / sample_num
            else:
                raise ValueError('Invalid weight method: %s!' % self.weight_srategy)
        else:
            old_weights = list()
            for i, r in enumerate(budget_list):
                _weight = self.weighted_surrogate.surrogate_weight[r]
                old_weights.append(_weight)
            new_weights = old_weights.copy()

        self.logger.info('[%s] %d-th Updating weights: %s' % (
            self.weight_srategy, self.weight_changed_cnt, str(new_weights)))

        # Assign the weight to each basic surrogate.
        for i, r in enumerate(budget_list):
            self.weighted_surrogate.surrogate_weight[r] = new_weights[i]
        self.weight_changed_cnt += 1
        # Save the weight data.
        # self.hist_weights.append(new_weights)
        # dir_path = os.path.join(self.data_directory, 'saved_weights')
        # file_name = 'mfes_weights_%s.npy' % (self.method_name,)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        # np.save(os.path.join(dir_path, file_name), np.asarray(self.hist_weights))
        # self.logger.info('update_weight() cost %.2fs. new weights are saved to %s'
        #                  % (time.time()-start_time, os.path.join(dir_path, file_name)))
    def _suggest(self, n_suggestions=1):
        trial_list = []
        trials_budgets = np.array([float(info[Key.BUDGET]) for info in self.trials.infos])

        # currently only suggest one
        if (self.trials.trials_num) < self.init_budget :
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
        elif self.rng.rand() < self.random_fraction or (trials_budgets==self.all_budgets[-1]).sum() == 0:
            configs = self.space.sample_configuration(n_suggestions)
            for config in configs:
                trial_list.append(
                    Trial(configuration=config,
                          config_dict=config.get_dictionary(),
                          array=config.get_array(sparse=False)))
        else:
            # update target surrogate model
            X_all = np.asarray(self.trials.get_array())
            y_all = np.asarray(self.trials.get_history()[0])
            for budget in self.all_budgets:
                mask = trials_budgets == budget
                self.weighted_surrogate.train(
                    X_all[mask],
                    std_normalization(y_all[mask]),budget)
            # calculate base incuments (only use for acq base EI)
            std_incumbent_value = np.min(std_normalization(self.target_y[self.all_budgets[-1]]))

            self.acquisition_func.update(surrogate_model=self.weighted_surrogate,
                                         y_best=std_incumbent_value)
            # caculate weight for base+target model
            # weight = self.weighted_surrogate.get_weight(self.trials)
            # self.weighted_surrogate.update_weight(weight)
            # self.acquisition_func.update_weight(weight)
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
            job_info = trial.info
            budget = job_info[Key.BUDGET]
            self.target_x[budget].append(trial.array)
            self.target_y[budget].append(trial.observe_value)

class MFES_BOHB_CG(BasicConfigGenerator):
    def __init__(self, sub_opt, cs, budget, max_pop_size, rng,
                 **kwargs) -> None:
        BasicConfigGenerator.__init__(self, cs, budget, max_pop_size, rng,
                                      **kwargs)
        self.sub_opt = sub_opt
        self.reset(max_pop_size)
    def _suggest(self, *args, **kwargs):
        return self.sub_opt._suggest(*args, **kwargs)

    def _observe(self, *args, **kwargs):
        return self.sub_opt._observe(*args, **kwargs)

    def update_weight(self, *args, **kwargs):
        return self.sub_opt.update_weight(*args, **kwargs)

alg_marker = 'mfes-bohb'
@alg_register.register(alg_marker)
class MFES_BOHB(BOHB):
    name = alg_marker

    def __init__(
            self,
            space: DenseConfigurationSpace,
            budget_bound=[9, 729],
            #  mutation_factor=0.5,
            #  crossover_prob=0.5,
            #  strategy='rand1_bin',
            eta: int = 3,
            seed: int = 42,
            round_limit: int = 1,
            bracket_limit=np.inf,
            boundary_fix_type='random',
            **kwargs):
        BOHB.__init__(self,
                      space,
                      budget_bound,
                      eta,
                      seed=seed,
                      round_limit=round_limit,
                      bracket_limit=bracket_limit,
                      boundary_fix_type=boundary_fix_type,
                      **kwargs)


        self.seed = seed
        self.logger = logger
        # Parameter for weight method `rank_loss_p_norm`.
        # self.power_num = power_num
        # Specify the weight learning method.
        # self.weight_srategy = weight_srategy
        self.weight_update_id = 0
        self.weight_changed_cnt = 0

        

        # self.incumbent_configs = []
        # self.incumbent_perfs = []

        # self.iterate_id = 0
        # self.iterate_r = []
        # self.hist_weights = list()

        # Saving evaluation statistics in Hyperband.
        # self.target_x = dict()
        # self.target_y = dict()
        # for index, item in enumerate(
        #         np.logspace(0, self.s_max, self.s_max + 1, base=self.eta)):
        #     r = int(item)
        #     self.iterate_r.append(r)
        #     self.target_x[r] = []
        #     self.target_y[r] = []

        # self.sls_max_steps = None
        # self.n_sls_iterations = 5
        # self.sls_n_steps_plateau_walk = 10



    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.cg = {}
        # types, bounds = get_types(self.space)

        sub_opt = SMBO(space=self.space,
                                all_budgets=self.budgets.tolist(),
                                weight_srategy='rank_loss_p_norm',
                                surrogate='rf',init_budget=0,
                                **kwargs)

        for i, b in enumerate(self._max_pop_size.keys()):
            self.cg[b] = MFES_BOHB_CG(sub_opt,
                                      self.space,
                                      seed=self.rng.randint(MAXINT),
                                      budget=b,
                                      max_pop_size=self._max_pop_size[b],
                                      rng=self.rng,
                                      **kwargs)

    def _acquire_candidate(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        # select a parent/target
        parent_id = self._get_next_idx_for_subpop(budget, bracket)

        if budget != bracket.budgets[0]:
            if bracket.is_new_rung():
                # TODO: check if generalizes to all budget spacings
                lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
                self._get_promotion_candidate(lower_budget, budget,
                                              num_configs)
            # else: # 每一列中的第一行，随机生成config

        else:
            if bracket.is_new_rung():
                lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
                self.cg[budget].population_fitness[:] = np.inf
                if self.round_recoder >= 1:
                    self.cg[budget].update_weight()

            # for b in reversed(self.budgets):
            #     if self.cg[b].kde_models:
            #         break
            # trial = self.cg[b]._suggest()[0]
            trials = self.cg[budget]._suggest(1)
            for i in range(len(trials)):
                self.cg[budget].population[parent_id + i] = trials[i].array
                # self.cg[budget].population[parent_id] = trial.array
        # parent_id = self._get_next_idx_for_subpop(budget, bracket)

        target = self.cg[budget].population[parent_id]
        # target = self.fix_boundary(target)
        return target, parent_id

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, True)
            fitness = trial.observe_value
            job_info = trial.info
            # learner_train_time = job_info.get(Key.EVAL_TIME, 0)
            budget = job_info[Key.BUDGET]
            parent_id = job_info['parent_id']
            individual = trial.array  # TODO
            for bracket in self.active_brackets:
                if bracket.bracket_id == job_info['bracket_id']:
                    # registering is IMPORTANT for Bracket Manager to perform SH
                    bracket.register_job(budget)  # may be new row
                    # bracket job complete
                    bracket.complete_job(
                        budget)  # IMPORTANT to perform synchronous SH
            self.cg[budget].population[parent_id] = individual
            self.cg[budget].population_fitness[parent_id] = fitness
            # updating incumbents
            if fitness < self.current_best_fitness:
                self.current_best = individual
                self.current_best_fitness = trial.observe_value
                self.current_best_trial = trial
        # for rung in range(bracket.n_rungs - 1, bracket.current_rung, -1):
        #     if self.cg[bracket.budgets[rung]].kde_models:
        #         break
        # else:
            self.cg[budget]._observe([trial])

        self._clean_inactive_brackets()

    def _sample_nonduplicate_config(self, num_configs=1):
        configs = list()
        sample_cnt = 0
        max_sample_cnt = 1000
        while len(configs) < num_configs:
            config = self.space.sample_configuration()[0]
            sample_cnt += 1
            if (not self.trials.is_contain(config)) and config not in configs:
                configs.append(config)
                sample_cnt = 0
                continue
            if sample_cnt >= max_sample_cnt:
                logger.warning(
                    'Cannot sample non duplicate configuration after %d iterations.'
                    % max_sample_cnt)
                configs.append(config)
                sample_cnt = 0
        return configs

opt_class = BOHB
