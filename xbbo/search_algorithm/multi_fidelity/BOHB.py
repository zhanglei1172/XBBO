'''
Reference: https://github.com/automl/DEHB
'''

import time
from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.multi_fidelity.hyperband import HB
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import DEHB_ConfigGenerator, SHBracketManager
from .. import alg_register


@alg_register.register('bohb')
class BOHB(HB):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 budget_bound=[9, 729],
                 mutation_factor=0.5,
                 crossover_prob=0.5,
                 strategy='rand1_bin',
                 eta: int = 3,
                 seed: int = 42,
                 round_limit: int = 1,
                 boundary_fix_type='random',
                 **kwargs):
        HB.__init__(self,
                    space,
                    budget_bound,
                    eta,
                    seed=seed,
                    round_limit=round_limit,
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

        self.round_limit = round_limit
        self.round_recoder = -1
        self.learner_time_recoder = 0
        self.total_time_recoder = 0
        self.kwargs = kwargs
        self._get_max_pop_sizes()
        self._init_subpop(**kwargs)


    def _suggest(self, n_suggestions=1):
        st = time.time()
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
                          "budget": budget,
                          "parent_id": parent_id,
                          "bracket_id": bracket.bracket_id
                      },
                      origin='HB'))
        self.total_time_recoder += time.time() - st
        return trial_list

    def _observe(self, trial_list):
        st = time.time()
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            fitness = trial.observe_value
            job_info = trial.info
            learner_train_time = job_info.get('eval_time', 0)
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
            if fitness <= self.de[budget].population_fitness[parent_id]:  # TODO
                self.de[budget].population[parent_id] = individual
                self.de[budget].population_fitness[parent_id] = fitness
                # updating incumbents
                if fitness < self.current_best_fitness:
                    self.current_best = individual
                    self.current_best_fitness = trial.observe_value
                    self.current_best_trial = trial

        self._clean_inactive_brackets()
        if len(self.active_brackets) == 0: # complete current bracket
            self.round_recoder = self.bracket_counter // self.max_SH_iter
        self.total_time_recoder += time.time() - st + learner_train_time
        self.learner_time_recoder += learner_train_time



    def _init_subpop(self, **kwargs):
        """ List of DE objects corresponding to the budgets (fidelities)
        """
        self.de = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.de[b] = DEHB_ConfigGenerator(
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
        target = self.de[budget].population[parent_id]
        lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
        # Fix bugs in the original author's implementation code
        if self.bracket_counter == 0 and budget != bracket.budgets[0]:
            # 第一列，第二行开始，直接挑选最优的进入下一轮
            # TODO: check if generalizes to all budget spacings
            individual = self._get_promotion_candidate(lower_budget, budget,
                                            num_configs)
            individual = self.fix_boundary(individual)
            return individual, parent_id
            # else: # 每一列中的第一行，随机生成config
        mutation_pop_idx = np.argsort(self.de[lower_budget].population_fitness)[:num_configs]
        mutation_pop = self.de[lower_budget].population[mutation_pop_idx]
        # generate mutants from previous budget subpopulation or global population
        if len(mutation_pop) < self.de[budget]._min_pop_size:
            filler = self.de[budget]._min_pop_size - len(mutation_pop) + 1
            new_pop = self.de[budget]._init_mutant_population(
                pop_size=filler, population=self._concat_pops(),
                target=target, best=self.current_best
            )
            mutation_pop = np.concatenate((mutation_pop, new_pop))
        mutant = self.de[budget].mutation(
            current=target, best=self.current_best, alt_pop=mutation_pop
        )
        # perform crossover with selected parent
        individual = self.de[budget].crossover(target=target, mutant=mutant)

        individual = self.fix_boundary(individual)
        return individual, parent_id

    def _get_promotion_candidate(self, low_budget, high_budget, n_configs):
        """ Manages the population to be promoted from the lower to the higher budget.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        evaluated_configs = np.where(self.de[low_budget].population_fitness != np.inf)[0]
        promotion_candidate_pop = self.de[low_budget].population[evaluated_configs]
        promotion_candidate_fitness = self.de[low_budget].population_fitness[evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)

        # creating population for promotion if none promoted yet or nothing to promote
        if self.de[high_budget].promotion_pop is None or \
                len(self.de[high_budget].promotion_pop) == 0:
            self.de[high_budget].promotion_pop = np.empty((0, self.dimension))
            self.de[high_budget].promotion_fitness = np.array([])

            # iterating over the evaluated individuals from the lower budget and including them
            # in the promotion population for the higher budget only if it's not in the population
            # this is done to ensure diversity of population and avoid redundant evaluations
            for idx in pop_idx:
                individual = promotion_candidate_pop[idx]
                # checks if the candidate individual already exists in the high budget population
                if np.any(np.all(individual == self.de[high_budget].population, axis=1)):
                    # skipping already present individual to allow diversity and reduce redundancy
                    continue
                self.de[high_budget].promotion_pop = np.append(
                    self.de[high_budget].promotion_pop, [individual], axis=0
                )
                self.de[high_budget].promotion_fitness = np.append(
                    self.de[high_budget].promotion_pop, promotion_candidate_fitness[pop_idx]
                )
            # retaining only n_configs
            self.de[high_budget].promotion_pop = self.de[high_budget].promotion_pop[:n_configs]
            self.de[high_budget].promotion_fitness = \
                self.de[high_budget].promotion_fitness[:n_configs]

        if len(self.de[high_budget].promotion_pop) > 0:
            config = self.de[high_budget].promotion_pop[0]
            # removing selected configuration from population
            self.de[high_budget].promotion_pop = self.de[high_budget].promotion_pop[1:]
            self.de[high_budget].promotion_fitness = self.de[high_budget].promotion_fitness[1:]
        else:
            # in case of an edge failure case where all high budget individuals are same
            # just choose the best performing individual from the lower budget (again)
            config = self.de[low_budget].population[pop_idx[0]]
        return config

    def _get_next_idx_for_subpop(self, budget, bracket):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.de[budget].parent_idx
        self.de[budget].parent_idx += 1
        self.de[budget].parent_idx = self.de[
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
            pop.extend(self.de[budget].population.tolist())
        return np.array(pop)



opt_class = BOHB
