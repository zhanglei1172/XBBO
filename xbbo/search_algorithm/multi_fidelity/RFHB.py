
from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import RFHB_ConfigGenerator, SHBracketManager
from xbbo.utils.constants import Key
from .. import alg_register

alg_marker = 'rfhb'

@alg_register.register(alg_marker)
class RFHB(HB):
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
        HB.__init__(self,
                    space,
                    budget_bound,
                    eta,
                    seed=seed,
                    round_limit=round_limit,
                    bracket_limit=bracket_limit,
                    boundary_fix_type=boundary_fix_type,
                    **kwargs)

    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.cg = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.cg[b] = RFHB_ConfigGenerator(
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
            self.cg[budget].reset(bracket.current_n_config)
            self.cg[budget].population_fitness[:] = np.inf
            if budget != bracket.budgets[0]:  # 每一列中的第二行开始，进行seed
                # TODO: check if generalizes to all budget spacings
                
                self._get_promotion_candidate(lower_budget, budget,
                                              num_configs)
            else:
            # else: # 每一列中的第一行，随机生成config,或者使用随机森林拒绝采样
                self.cg[budget].update_new_subpopulation(self.cg[budget]._init_population(num_configs))

        parent_id = self._get_next_idx_for_subpop(budget, bracket)
        target = self.cg[budget].population[parent_id]

        target = self.fix_boundary(target)
        return target, parent_id

    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        evaluated_configs = np.where(
            self.cg[low_budget].population_fitness != np.inf)[0]
        promotion_candidate_pop = self.cg[low_budget].population[
            evaluated_configs]
        promotion_candidate_fitness = self.cg[low_budget].population_fitness[
            evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)
        labels = promotion_candidate_fitness.copy()
        # labels = np.zeros_like(promotion_candidate_fitness)
        elite_index = pop_idx[:n_configs]

        # labels[elite_index] = 1
        # fiting low; gen high
        self.cg[low_budget].add_fitting(promotion_candidate_pop, labels)
        # promotion top to down
        self.cg[high_budget].update_new_subpopulation(
            self.cg[low_budget].population[elite_index])

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
            if fitness <= self.cg[budget].population_fitness[parent_id]:  # TODO
                self.cg[budget].population[parent_id] = individual
                self.cg[budget].population_fitness[parent_id] = fitness
                # updating incumbents
                if fitness < self.current_best_fitness:
                    self.current_best = individual
                    self.current_best_fitness = trial.observe_value
                    self.current_best_trial = trial

        self._clean_inactive_brackets()
        # if len(self.active_brackets) == 0:  # complete current bracket
        #     self.round_recoder = self.bracket_counter // self.max_SH_iter

opt_class = RFHB
