'''
Reference: https://github.com/automl/DEHB
'''

import time
from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import ConfigGenerator, SHBracketManager
from .. import alg_register


@alg_register.register('hb')
class HB(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 budget_bound=[9, 729],
                 eta: int = 3,
                 seed: int = 42,
                 round_limit: int = 1,
                 bracket_limit: int = np.inf,
                 boundary_fix_type='random',
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='round',
                                   encoding_ord='round',
                                   seed=seed,
                                   **kwargs)

        # Uniform2Gaussian.__init__(self,)
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()
        self.fix_type = boundary_fix_type
        # self.dimension = len(configs)
        # self.candidates = [None] * self.llambda
        self.trials = Trials(dim=self.dimension)
        self.current_best = None
        self.current_best_fitness = np.inf

        self.min_budget, self.max_budget = budget_bound
        self.eta = eta
        self.max_SH_iter = None
        self.budgets = None
        if self.min_budget is not None and \
           self.max_budget is not None and \
           self.eta is not None:
            self.max_SH_iter = -int(
                np.log(self.min_budget / self.max_budget) /
                np.log(self.eta)) + 1  # max_sh_iter是数量，0~t,(t+1)个
            self.budgets: List[int] = self.max_budget * np.power(
                self.eta, -np.linspace(
                    start=self.max_SH_iter - 1, stop=0, num=self.max_SH_iter))
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects

        self.round_limit = round_limit
        self.bracket_limit = bracket_limit
        self.bracket_counter = -1 # completed bracket count
        self.round_recoder = -1 # completed cound count
        self.learner_time_recoder = 0
        self.total_time_recoder = 0
        self.kwargs = kwargs
        self._get_max_pop_sizes()
        self._init_subpop(**kwargs)

    def check_stop(self, ):
        if AbstractOptimizer.check_stop(self) or self.round_recoder >= self.round_limit or self.bracket_counter >= self.bracket_limit : # FIXME
            return True
        else:
            return False

    def _start_new_bracket(self):
        """ Starts a new bracket based on Hyperband
        """
        # start new bracket
        # self.round_recoder = self.bracket_counter // self.max_SH_iter
        self.bracket_counter += 1  # iteration counter gives the bracket count or bracket ID
        n_configs, budgets = self._get_next_bracket_space(self.bracket_counter)
        bracket = SHBracketManager(n_configs=n_configs,
                                   budgets=budgets,
                                   bracket_id=self.bracket_counter)
        self.active_brackets.append(bracket)
        return bracket

    def _get_next_bracket_space(self, iteration):
        '''Computes the Successive Halving spacing

        Given the iteration index, computes the budget spacing to be used and
        the number of configurations to be used for the SH iterations.

        Parameters
        ----------
        iteration : int
            Iteration index
        clip : int, {1, 2, 3, ..., None}
            If not None, clips the minimum number of configurations to 'clip'

        Returns
        -------
        ns : array
        budgets : array
        '''
        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # budget spacing for this iteration
        budgets = self.budgets[(-s - 1):]
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta**s)
        ns = [max(int(n0 * (self.eta**(-i))), 1) for i in range(s + 1)]
        # if self.min_clip is not None and self.max_clip is not None:
        #     ns = np.clip(ns, a_min=self.min_clip, a_max=self.max_clip)
        # elif self.min_clip is not None:
        #     ns = np.clip(ns, a_min=self.min_clip, a_max=np.max(ns))

        return ns, budgets

    def _suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            if len(self.active_brackets) == 0 or \
                np.all([bracket.is_bracket_done() for bracket in self.active_brackets]):
                bracket = self._start_new_bracket()
            else:
                for _bracket in self.active_brackets:
                    # check if _bracket is not waiting for previous rung results of same bracket
                    # _bracket is not waiting on the last rung results
                    # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                    if not _bracket.previous_rung_waits(
                    ) and _bracket.is_pending():
                        # bracket eligible for job scheduling
                        bracket = _bracket
                        break
                if bracket is None:
                    # start new bracket when existing list has all waiting brackets
                    bracket = self._start_new_bracket()  # new column

                    # budget that the SH bracket allots
            budget = bracket.get_next_job_budget()
            # new individual
            candidate, parent_id = self._acquire_candidate(bracket, budget)
            # notifies the Bracket Manager that a single config is to run for the budget chosen
            # job_info = {
            #     "config": config,
            #     "budget": budget,
            #     "parent_id": parent_id,
            #     "bracket_id": bracket.bracket_id
            # }
            # # return job_info

            config = DenseConfiguration.from_array(self.space, candidate)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=candidate,
                      info={
                          "budget": budget,
                          "parent_id": parent_id,
                          "bracket_id": bracket.bracket_id
                      },
                      origin='HB'))
        return trial_list

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            fitness = trial.observe_value
            job_info = trial.info
            # learner_train_time = job_info.get('eval_time', 0)
            budget = job_info['budget']
            parent_id = job_info['parent_id']
            individual = trial.array  # TODO
            for bracket in self.active_brackets:
                if bracket.bracket_id == job_info['bracket_id']:
                    # registering is IMPORTANT for Bracket Manager to perform SH
                    bracket.register_job(budget)  # may be new row
                    # bracket job complete
                    bracket.complete_job(
                        budget)  # IMPORTANT to perform synchronous SH
            # carry out DE selection
            assert np.isposinf(self.de[budget].population_fitness[parent_id])
            if fitness <= self.de[budget].population_fitness[parent_id]:  # TODO
                self.de[budget].population[parent_id] = individual
                self.de[budget].population_fitness[parent_id] = fitness
                # updating incumbents
                if fitness < self.current_best_fitness:
                    self.current_best = individual
                    self.current_best_fitness = trial.observe_value
                    self.current_best_trial = trial


        self._clean_inactive_brackets()
        # if len(self.active_brackets) == 0: # complete current bracket
        #     self.round_recoder = self.bracket_counter // self.max_SH_iter
            # self._start_new_bracket()

    def _clean_inactive_brackets(self):
        """ Removes brackets from the active list if it is done as communicated by Bracket Manager
        """
        self.active_brackets = [
            bracket for bracket in self.active_brackets
            if ~bracket.is_bracket_done()
        ]
        if len(self.active_brackets) == 0:
            self._start_new_bracket()
            self.round_recoder = self.bracket_counter // self.max_SH_iter
        return

    def _get_max_pop_sizes(self):
        """Determines maximum pop size(config num) for each budget
        """
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self._get_next_bracket_space(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = max(
                    n[j], self._max_pop_size[r_j]
                ) if r_j in self._max_pop_size.keys() else n[j]

    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.de = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.de[b] = ConfigGenerator(self.space,
                                         budget=b,
                                         max_pop_size=self._max_pop_size[b],
                                         rng=self.rng,
                                         **kwargs)
            # self.de[b].population = self.de[b].init_population(
            #     pop_size=self._max_pop_size[b])
            # self.de[b].population_fitness = np.array([np.inf] * self._max_pop_size[b])
            # # adding attributes to DEHB objects to allow communication across subpopulations
            # self.de[b].parent_idx = 0
            # self.de[b].promotion_pop = None
            # self.de[b].promotion_fitness = None

    def _acquire_candidate(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        if bracket.is_new_rung():
            self.de[budget].reset(bracket.current_n_config)
            if budget != bracket.budgets[0]:  # 每一列中的第二行开始，进行seed
                # TODO: check if generalizes to all budget spacings
                lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
                self._get_promotion_candidate(lower_budget, budget,
                                              num_configs)
            # else: # 每一列中的第一行，随机生成config

        parent_id = self._get_next_idx_for_subpop(budget, bracket)

        target = self.de[budget].population[parent_id]

        target = self.fix_boundary(target)
        return target, parent_id

    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        evaluated_configs = np.where(
            self.de[low_budget].population_fitness != np.inf)[0]
        promotion_candidate_pop = self.de[low_budget].population[
            evaluated_configs]
        promotion_candidate_fitness = self.de[low_budget].population_fitness[
            evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)
        # n_configs = len(self.de[high_budget].population) # not sure eta multiple configs
        elite_index = pop_idx[:n_configs]
        # self.de[high_budget].population_fitness = self.de[low_budget].population_fitness[elite_index]
        self.de[high_budget].population = self.de[low_budget].population[
            elite_index]
        # return

    def _get_next_idx_for_subpop(self, budget, bracket):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.de[budget].parent_idx
        self.de[budget].parent_idx += 1
        # self.de[budget].parent_idx = self.de[
        #     budget].parent_idx % self._max_pop_size[budget]
        self.de[budget].parent_idx = self.de[
            budget].parent_idx % bracket.current_n_config
        return parent_id


opt_class = HB
