'''
Reference: https://github.com/automl/DEHB
'''

from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import RFHB_ConfigGenerator, SHBracketManager
from xbbo.utils.constants import Key
from .. import alg_register


@alg_register.register('rfhb')
class RFHB(HB):
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
        HB.__init__(self,
                    space,
                    budget_bound,
                    eta,
                    seed=seed,
                    round_limit=round_limit,
                    bracket_limit=bracket_limit,
                    boundary_fix_type=boundary_fix_type,
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
        self.bracket_counter = -1
        self._max_pop_size = None
        self.active_brackets = []  # list of SHBracketManager objects

        # self.round_limit = round_limit
        # self.round_recoder = -1

        self.kwargs = kwargs
        self._get_max_pop_sizes()
        self._init_subpop(**kwargs)

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

            config = DenseConfiguration.from_array(self.space, candidate)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=candidate,
                      info={
                          Key.BUDGET: budget,
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
            # carry out DE selection
            if fitness <= self.rf[budget].population_fitness[parent_id]:  # TODO
                self.rf[budget].population[parent_id] = individual
                self.rf[budget].population_fitness[parent_id] = fitness
                # updating incumbents
                if fitness < self.current_best_fitness:
                    self.current_best = individual
                    self.current_best_fitness = trial.observe_value
                    self.current_best_trial = trial
            else:
                print(1)
        self._clean_inactive_brackets()
        # if len(self.active_brackets) == 0:  # complete current bracket
        #     self.round_recoder = self.bracket_counter // self.max_SH_iter

    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.rf = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.rf[b] = RFHB_ConfigGenerator(
                self.space,
                budget=b,
                max_pop_size=self._max_pop_size[b],
                rng=self.rng,
                eta = self.eta,
                **kwargs)

    def _acquire_candidate(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        if bracket.is_new_rung():
            lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
            self.rf[budget].reset(bracket.current_n_config)
            if budget != bracket.budgets[0]:  # 每一列中的第二行开始，进行seed
                # TODO: check if generalizes to all budget spacings
                
                self._get_promotion_candidate(lower_budget, budget,
                                              num_configs)
            else:
            # else: # 每一列中的第一行，随机生成config,或者使用随机森林拒绝采样
                self.rf[budget].update_new_subpopulation(self.rf[budget]._init_population(num_configs))

        parent_id = self._get_next_idx_for_subpop(budget, bracket)
        target = self.rf[budget].population[parent_id]

        target = self.fix_boundary(target)
        return target, parent_id

    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        evaluated_configs = np.where(
            self.rf[low_budget].population_fitness != np.inf)[0]
        promotion_candidate_pop = self.rf[low_budget].population[
            evaluated_configs]
        promotion_candidate_fitness = self.rf[low_budget].population_fitness[
            evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)
        labels = promotion_candidate_fitness.copy()
        # labels = np.zeros_like(promotion_candidate_fitness)
        elite_index = pop_idx[:n_configs]

        # labels[elite_index] = 1
        # fiting low; gen high
        self.rf[low_budget].add_fitting(promotion_candidate_pop, labels)
        self.rf[high_budget].update_new_subpopulation(
            self.rf[low_budget].population[elite_index])

    def _get_next_idx_for_subpop(self, budget, bracket):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.rf[budget].parent_idx
        self.rf[budget].parent_idx += 1
        # self.de[budget].parent_idx = self.de[
        #     budget].parent_idx % self._max_pop_size[budget]
        self.rf[budget].parent_idx = self.rf[
            budget].parent_idx % bracket.current_n_config
        return parent_id


opt_class = RFHB
