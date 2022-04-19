'''
Reference: https://github.com/automl/DEHB
'''

from collections import defaultdict
from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import BasicConfigGenerator, DEHB_ConfigGenerator, SHBracketManager
# from xbbo.search_algorithm.de_optimizer import DE
from xbbo.utils.constants import Key
from .. import alg_register

# class DEHB_CG(BasicConfigGenerator, DE):
#     def __init__(self, cs, budget, max_pop_size, rng, **kwargs) -> None:
#         BasicConfigGenerator.__init__(self, cs, budget, max_pop_size, rng, **kwargs)
#         DE.__init__(self, space=cs, init_budget=0,**kwargs)
#         self.reset(max_pop_size)

alg_marker = 'dehb'


@alg_register.register(alg_marker)
class DEHB(HB):
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
            encoding_cat='bin',
            encoding_ord='bin',
            **kwargs):
        HB.__init__(self,
                    space,
                    budget_bound,
                    eta,
                    seed=seed,
                    round_limit=round_limit,
                    bracket_limit=bracket_limit,
                    boundary_fix_type=boundary_fix_type,
                    encoding_cat=encoding_cat,
                    encoding_ord=encoding_ord,
                    **kwargs)
        self.exp_selection_success = {k:[] for k in self.cg}
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
                self.exp_selection_success[budget].append(1)
            else:
                self.exp_selection_success[budget].append(0)

        self._clean_inactive_brackets()
        # if len(self.active_brackets) == 0:  # complete current bracket
        #     self.round_recoder = self.bracket_counter // self.max_SH_iter

    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.cg = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.cg[b] = DEHB_ConfigGenerator(
                self.space,
                budget=b,
                max_pop_size=self._max_pop_size[b],
                rng=self.rng,
                **kwargs)

    def _acquire_candidate(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        # select a parent/target
        parent_id = self._get_next_idx_for_subpop(budget, bracket)
        target = self.cg[budget].population[parent_id]
        lower_budget, num_configs = bracket.get_lower_budget_promotions(budget)
        # if bracket.is_new_rung():
        #     self.cg[budget].population_fitness[num_configs:] = np.inf
        # Fix bugs in the original author's implementation code
        # if self.bracket_counter == 0 and budget != bracket.budgets[0]:
        if self.bracket_counter < self.max_SH_iter and budget != bracket.budgets[
                0]:
            # 第一轮HB，第二行开始，直接挑选最优的进入下一轮
            # TODO: check if generalizes to all budget spacings
            individual = self._get_promotion_candidate(lower_budget, budget,
                                                       num_configs)
            individual = self.fix_boundary(individual)
            return individual, parent_id
            # else: # 每一列中的第一行，随机生成config
        mutation_pop_idx = np.argsort(
            self.cg[lower_budget].population_fitness)[:num_configs]
        # mutation_pop_idx = np.argpartition(self.cg[lower_budget].population_fitness, num_configs-1)[:num_configs]
        mutation_pop = self.cg[lower_budget].population[mutation_pop_idx]
        # generate mutants from previous budget subpopulation or global population
        if len(mutation_pop) < self.cg[budget]._min_pop_size:
            filler = self.cg[budget]._min_pop_size - len(mutation_pop) + 1
            new_pop = self.cg[budget]._init_mutant_population(
                pop_size=filler,
                population=self._concat_pops(),
                target=target,
                best=self.current_best)
            mutation_pop = np.concatenate((mutation_pop, new_pop))
        mutant = self.cg[budget].mutation(current=target,
                                          best=self.current_best,
                                          alt_pop=mutation_pop)
        # perform crossover with selected parent
        individual = self.cg[budget].crossover(target=target, mutant=mutant)

        individual = self.fix_boundary(individual)
        return individual, parent_id

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

        # creating population for promotion if none promoted yet or nothing to promote
        if self.cg[high_budget].promotion_pop is None or \
                len(self.cg[high_budget].promotion_pop) == 0:
            self.cg[high_budget].promotion_pop = np.empty((0, self.dimension))
            self.cg[high_budget].promotion_fitness = np.array([])

            # iterating over the evaluated individuals from the lower budget and including them
            # in the promotion population for the higher budget only if it's not in the population
            # this is done to ensure diversity of population and avoid redundant evaluations
            for idx in pop_idx:
                individual = promotion_candidate_pop[idx]
                # checks if the candidate individual already exists in the high budget population
                if np.any(
                        np.all(individual == self.cg[high_budget].population,
                               axis=1)):
                    # skipping already present individual to allow diversity and reduce redundancy
                    continue
                self.cg[high_budget].promotion_pop = np.append(
                    self.cg[high_budget].promotion_pop, [individual], axis=0)
                self.cg[high_budget].promotion_fitness = np.append(
                    self.cg[high_budget].promotion_pop,
                    promotion_candidate_fitness[pop_idx])
            # retaining only n_configs
            self.cg[high_budget].promotion_pop = self.cg[
                high_budget].promotion_pop[:n_configs]
            self.cg[high_budget].promotion_fitness = \
                self.cg[high_budget].promotion_fitness[:n_configs]

        if len(self.cg[high_budget].promotion_pop) > 0:
            config = self.cg[high_budget].promotion_pop[0]
            # removing selected configuration from population
            self.cg[high_budget].promotion_pop = self.cg[
                high_budget].promotion_pop[1:]
            self.cg[high_budget].promotion_fitness = self.cg[
                high_budget].promotion_fitness[1:]
        else:
            # in case of an edge failure case where all high budget individuals are same
            # just choose the best performing individual from the lower budget (again)
            config = self.cg[low_budget].population[pop_idx[0]]
        return config

    def _get_next_idx_for_subpop(self, budget, bracket):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.cg[budget].parent_idx
        self.cg[budget].parent_idx += 1
        self.cg[budget].parent_idx = self.cg[
            budget].parent_idx % self._max_pop_size[budget]
        return parent_id

    def _concat_pops(self, exclude_budget=None):
        """ Concatenates all subpopulations
        """
        budgets = list(self.budgets)
        if exclude_budget is not None:
            budgets.remove(exclude_budget)
        pop = []
        for budget in budgets:
            pop.extend(self.cg[budget].population.tolist())
        return np.array(pop)


opt_class = DEHB
