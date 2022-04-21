'''
Reference: https://github.com/automl/DEHB
'''

from typing import List
import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from xbbo.search_algorithm.multi_fidelity.utils.bracket_manager import BasicConfigGenerator, ConfigGenerator, SHBracketManager
from xbbo.search_algorithm.random_optimizer import RandomOptimizer

from xbbo.utils.constants import Key
from .. import alg_register

class HB_CG(BasicConfigGenerator, RandomOptimizer):
    def __init__(self, cs, budget, max_pop_size, rng, **kwargs) -> None:
        BasicConfigGenerator.__init__(self, cs, budget, max_pop_size, rng, **kwargs)
        RandomOptimizer.__init__(self, space=cs, init_budget=0, **kwargs)
        self.reset(max_pop_size)

alg_marker = 'hb'


@alg_register.register('hb')
class HB(AbstractOptimizer):
    name = alg_marker
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
                                #    encoding_cat='bin',
                                #    encoding_ord='bin',
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
            #     Key.BUDGET: budget,
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
                          Key.BUDGET: budget,
                          "parent_id": parent_id,
                          "bracket_id": bracket.bracket_id
                      },
                      origin=self.name))
        return trial_list

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
        self.cg[budget]._observe(trial_list)


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
        self.cg = {}
        for i, b in enumerate(self._max_pop_size.keys()):
            self.cg[b] = HB_CG(self.space,
                                         budget=b,
                                         max_pop_size=self._max_pop_size[b],
                                         rng=self.rng,
                                         **kwargs)
            # self.cg[b].population = self.cg[b].init_population(
            #     pop_size=self._max_pop_size[b])
            # self.cg[b].population_fitness = np.array([np.inf] * self._max_pop_size[b])
            # # adding attributes to DEHB objects to allow communication across subpopulations
            # self.cg[b].parent_idx = 0
            # self.cg[b].promotion_pop = None
            # self.cg[b].promotion_fitness = None

    def _acquire_candidate(self, bracket, budget):
        """ Generates/chooses a configuration based on the budget and iteration number
        """
        parent_id = self._get_next_idx_for_subpop(budget, bracket)

        if budget != bracket.budgets[0]:
            if bracket.is_new_rung():
                # TODO: check if generalizes to all budget spacings
                lower_budget, num_configs = bracket.get_lower_budget_promotions(
                    budget)
                self._get_promotion_candidate(lower_budget, budget,
                                              num_configs)
            # else: # 每一列中的第一行，随机生成config
            target = self.cg[budget].population[parent_id]

        else: # config generator
            trial = self.cg[budget]._suggest()[0]

            target = trial.array
        # parent_id = self._get_next_idx_for_subpop(budget, bracket)

        # target = self.fix_boundary(target)
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
        # n_configs = len(self.cg[high_budget].population) # not sure eta multiple configs
        elite_index = pop_idx[:n_configs]
        # self.cg[high_budget].population_fitness = self.cg[low_budget].population_fitness[elite_index]
        self.cg[high_budget].population[:len(elite_index)] = promotion_candidate_pop[
            elite_index]
        # self.cg[high_budget].population_fitness[len(elite_index):] = np.inf
        self.cg[high_budget].population_fitness[:] = np.inf
        # return

    def _get_next_idx_for_subpop(self, budget, bracket):
        """ Maintains a looping counter over a subpopulation, to iteratively select a parent
        """
        parent_id = self.cg[budget].parent_idx
        self.cg[budget].parent_idx += 1
        # self.cg[budget].parent_idx = self.cg[
        #     budget].parent_idx % self._max_pop_size[budget]
        self.cg[budget].parent_idx = self.cg[
            budget].parent_idx % bracket.current_n_config
        return parent_id


opt_class = HB
