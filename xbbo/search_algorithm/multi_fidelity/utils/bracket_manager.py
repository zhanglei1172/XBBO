'''
Reference: https://github.com/automl/DEHB
'''

from typing import List
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from xbbo.utils.constants import Key


class SHBracketManager(object):
    """ Synchronous Successive Halving utilities
    """
    def __init__(self, n_configs, budgets, bracket_id=None):
        assert len(n_configs) == len(budgets)
        self.n_configs = n_configs
        self.budgets = budgets
        self.bracket_id = bracket_id
        self.pending_sh_bracket = {}
        self.done_sh_bracket = {}
        self._config_map = {}
        for i, budget in enumerate(budgets):
            # sh_bracket keeps track of jobs/configs that are still to be scheduled/allocatted
            # _sh_bracket keeps track of jobs/configs that have been run and results retrieved for
            # (sh_bracket[i] + _sh_bracket[i]) == n_configs[i] is when no jobs have been scheduled
            #   or all jobs for that budget/rung are over
            # (sh_bracket[i] + _sh_bracket[i]) < n_configs[i] indicates a job has been scheduled
            #   and is queued/running and the bracket needs to be paused till results are retrieved
            self.pending_sh_bracket[budget] = n_configs[
                i]  # each scheduled job does -= 1
            self.done_sh_bracket[budget] = 0  # each retrieved job does +=1
        self.n_rungs = len(budgets)
        self.current_rung = 0

    def is_new_rung(self,): # is new row?
        return self.pending_sh_bracket[self.get_budget()] == self.current_n_config
    @property
    def current_n_config(self,):
        return self.n_configs[self.current_rung]

    def get_budget(self, rung=None):
        """ Returns the exact budget that rung is pointing to.

        Returns current rung's budget if no rung is passed.
        """
        if rung is not None:
            return self.budgets[rung]
        return self.budgets[self.current_rung]

    def get_lower_budget_promotions(self, budget):
        """ Returns the immediate lower budget and the number of configs to be promoted from there
        """
        assert budget in self.budgets
        rung = np.where(budget == self.budgets)[0][0]
        prev_rung = np.clip(rung - 1, a_min=0, a_max=self.n_rungs - 1)
        lower_budget = self.budgets[prev_rung]
        num_promote_configs = self.n_configs[rung]
        return lower_budget, num_promote_configs

    def get_next_job_budget(self):
        """ Returns the budget that will be selected if current_rung is incremented by 1
        """
        if self.pending_sh_bracket[self.get_budget()] > 0:
            # the current rung still has unallocated jobs (>0)
            return self.get_budget()
        else:
            print('warnning')
            # the current rung has no more jobs to allocate, increment it
            rung = (self.current_rung + 1) % self.n_rungs
            if self.pending_sh_bracket[self.get_budget(rung)] > 0:
                # the incremented rung has unallocated jobs (>0)
                return self.get_budget(rung)
            else:
                # all jobs for this bracket has been allocated/bracket is complete
                # no more budgets to evaluate and can return None
                pass
            return None

    def register_job(self, budget):
        """ Registers the allocation of a configuration for the budget and updates current rung

        This function must be called when scheduling a job in order to allow the bracket manager
        to continue job and budget allocation without waiting for jobs to finish and return
        results necessarily. This feature can be leveraged to run brackets asynchronously.
        """
        assert budget in self.budgets
        assert self.pending_sh_bracket[budget] > 0
        self.pending_sh_bracket[budget] -= 1
        if not self._is_rung_pending(self.current_rung):
            # increment current rung if no jobs left in the rung
            self.current_rung = (self.current_rung + 1) % self.n_rungs

    def complete_job(self, budget):
        """ Notifies the bracket that a job for a budget has been completed

        This function must be called when a config for a budget has finished evaluation to inform
        the Bracket Manager that no job needs to be waited for and the next rung can begin for the
        synchronous Successive Halving case.
        """
        assert budget in self.budgets
        _max_configs = self.n_configs[list(self.budgets).index(budget)]
        assert self.done_sh_bracket[budget] < _max_configs
        self.done_sh_bracket[budget] += 1

    def _is_rung_waiting(self, rung):
        """ Returns True if at least one job is still pending/running and waits for results
        """
        job_count = self.done_sh_bracket[self.budgets[rung]] + self.pending_sh_bracket[
            self.budgets[rung]]
        if job_count < self.n_configs[rung]:
            return True
        return False

    def _is_rung_pending(self, rung):
        """ Returns True if at least one job pending to be allocatted in the rung
        """
        if self.pending_sh_bracket[self.budgets[rung]] > 0:
            return True
        return False

    def previous_rung_waits(self):
        """ Returns True if none of the rungs < current rung is waiting for results
        """
        for rung in range(self.current_rung):
            if self._is_rung_waiting(rung) and not self._is_rung_pending(rung):
                return True
        return False

    def is_bracket_done(self):
        """ Returns True if all configs in all rungs in the bracket have been allocated
        """
        return ~self.is_pending() and ~self.is_waiting()

    def is_pending(self):
        """ Returns True if any of the rungs/budgets have still a configuration to submit
        """
        return np.any(
            [self._is_rung_pending(i) > 0 for i, _ in enumerate(self.budgets)])

    def is_waiting(self):
        """ Returns True if any of the rungs/budgets have a configuration pending/running
        """
        return np.any(
            [self._is_rung_waiting(i) > 0 for i, _ in enumerate(self.budgets)])

    def __repr__(self):
        cell_width = 9
        cell = "{{:^{}}}".format(cell_width)
        budget_cell = "{{:^{}.2f}}".format(cell_width)
        header = "|{}|{}|{}|{}|".format(cell.format(Key.BUDGET),
                                        cell.format("pending"),
                                        cell.format("waiting"),
                                        cell.format("done"))
        _hline = "-" * len(header)
        table = [header, _hline]
        for i, budget in enumerate(self.budgets):
            pending = self.pending_sh_bracket[budget]
            done = self.done_sh_bracket[budget]
            waiting = np.abs(self.n_configs[i] - pending - done)
            entry = "|{}|{}|{}|{}|".format(budget_cell.format(budget),
                                           cell.format(pending),
                                           cell.format(waiting),
                                           cell.format(done))
            table.append(entry)
        table.append(_hline)
        return "\n".join(table)


class ConfigGenerator():
    def __init__(self, cs, budget, max_pop_size, rng, **kwargs) -> None:
        self.cs = cs
        self.budget = budget
        self.max_pop_size = max_pop_size
        self.rng = rng
        # self.fitness = np.array([np.inf] * self.pop_size)
        # self.reset()
        # self.parent_counter = 0


    def reset(self, pop_size):
        self.inc_score = np.inf
        self.inc_config = None
        self.population = self._init_population(pop_size)
        self.population_fitness = np.array([np.inf] * pop_size)
        # adding attributes to DEHB objects to allow communication across subpopulations
        self.parent_idx = 0
        self.promotion_pop = None
        self.promotion_fitness = None

        # self.history = []

    def _shuffle_pop(self):
        pop_order = np.arange(len(self.population))
        self.rng.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.population_fitness = self.population_fitness[pop_order]
        # self.age = self.age[pop_order]

    def _sort_pop(self):
        pop_order = np.argsort(self.population_fitness)
        self.rng.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.population_fitness = self.population_fitness[pop_order]
        self.age = self.age[pop_order]

    def _init_population(self, pop_size: int) -> List:
        # sample from ConfigSpace s.t. conditional constraints (if any) are maintained
        population = [config.get_array() for config in self.cs.sample_configuration(size=pop_size)]
        if not isinstance(population, List):
            population = [population]
        # the population is maintained in a list-of-vector form where each ConfigSpace
        return np.array(population)

class DEHB_ConfigGenerator(ConfigGenerator):
    def __init__(self, cs, budget, max_pop_size, rng, mutation_factor=0.5,crossover_prob=0.5,strategy='rand1_bin',**kwargs) -> None:
        super().__init__(cs, budget, max_pop_size, rng, **kwargs)
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.dimension = self.cs.get_dimensions()
        self.strategy = strategy
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.reset(max_pop_size)
        # self.population = [None] * self.llambda
        # self.population_fitness = [np.inf] * self.llambda
        # self.current_best = None
        # self.current_best_fitness = np.inf
        self._set_min_pop_size()

    def _set_min_pop_size(self):
        if self.mutation_strategy in ['rand1', 'rand2dir', 'randtobest1']:
            self._min_pop_size = 3
        elif self.mutation_strategy in ['currenttobest1', 'best1']:
            self._min_pop_size = 2
        elif self.mutation_strategy in ['best2']:
            self._min_pop_size = 4
        elif self.mutation_strategy in ['rand2']:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size


                
    def _mutation_rand1(self, r1, r2, r3):
        '''Performs the 'rand1' type of DE mutation
        '''
        
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def _mutation_rand2(self, r1, r2, r3, r4, r5):
        '''Performs the 'rand2' type of DE mutation
        '''
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def _mutation_currenttobest1(self, current, best, r1, r2):
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def _mutation_rand2dir(self, r1, r2, r3):
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation(self, current=None, best=None, alt_pop=None):
        '''Performs DE mutation
        '''
        if self.mutation_strategy == 'rand1':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop)

            mutant = self._mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == 'rand2':
            r1, r2, r3, r4, r5 = self._sample_population(size=5, alt_pop=alt_pop)
            mutant = self._mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == 'rand2dir':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop)

            mutant = self._mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == 'best1':
            r1, r2 = self._sample_population(size=2, alt_pop=alt_pop)

            if best is None:
                best = self.population[np.argmin(self.population_fitness)]
            mutant = self._mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == 'best2':
            r1, r2, r3, r4 = self._sample_population(size=4, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.population_fitness)]
            mutant = self._mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == 'currenttobest1':
            r1, r2 = self._sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.population_fitness)]
            mutant = self._mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == 'randtobest1':
            r1, r2, r3 = self._sample_population(size=3, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.population_fitness)]
            mutant = self._mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def _sample_population(self, size: int = 3, alt_pop= None):
        '''Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population (alt_pop)
        '''
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
                return np.array(self.population)[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                selection = self.rng.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return np.array(alt_pop)[selection]
        else:
            selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
            return np.array(self.population)[selection]

    def _crossover_bin(self, target, mutant):
        '''Performs the binomial crossover of DE
        '''
        cross_points = self.rng.rand(self.dimension) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.rng.randint(0, self.dimension)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def _crossover_exp(self, target, mutant):
        '''Performs the exponential crossover of DE
        '''
        n = self.rng.randint(0, self.dimension)
        L = 0
        while ((self.rng.rand() < self.crossover_prob) and L < self.dimension):
            idx = (n+L) % self.dimension
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        '''Performs DE crossover
        '''
        if self.crossover_strategy == 'bin':
            offspring = self._crossover_bin(target, mutant)
        elif self.crossover_strategy == 'exp':
            offspring = self._crossover_exp(target, mutant)
        return offspring


    def _init_mutant_population(self, pop_size, population, target=None, best=None):
        '''Generates pop_size mutants from the passed population
        '''
        mutants = np.zeros((pop_size, self.dimension))
        for i in range(pop_size):
            mutants[i] = self.mutation(current=target, best=best, alt_pop=population)
        return mutants


class RFHB_ConfigGenerator(ConfigGenerator):
    def __init__(self, cs, budget, max_pop_size, rng, eta=3, **kwargs) -> None:
        super().__init__(cs, budget, max_pop_size, rng, **kwargs)
        self.dimension = self.cs.get_dimensions()
        self.reset(max_pop_size)
        self.rf = RandomForestClassifier(n_estimators=25)
        self.reject_rate = kwargs.get("reject_rate", 10)
        self.trained_samples = np.empty((0,self.dimension))
        self.trained_labels = np.empty((0))
        self.fited = False
        self.eta = eta
        # self.population = [None] * self.llambda
        # self.population_fitness = [np.inf] * self.llambda
        # self.current_best = None
        # self.current_best_fitness = np.inf
        self._set_min_train_size()

    def _set_min_train_size(self):
        self._min_train_size = max(self.dimension, 10)

        return self._min_train_size


    def reset(self, pop_size):
        self.inc_score = np.inf
        self.inc_config = None
        self.parent_idx = 0
        self.population = self._init_population(pop_size)
        self.population_fitness = np.array([np.inf] * pop_size)
        # self.batch_sample_num = int(self.reject_rate * self.pop_size)
    def add_fitting(self, X, y):
        self.trained_samples = np.concatenate([self.trained_samples, X])
        self.trained_labels = np.concatenate([self.trained_labels, y])
        # labels = np.zeros_like(self.trained_labels)

        if len(self.trained_labels) >= self._min_train_size:
            tau = np.quantile(self.trained_labels, q=1/self.eta)
            labels = self.trained_labels <= tau
            self.rf.fit(self.trained_samples, labels)
            self.fited = True
    def update_new_subpopulation(self, elite_population):
        if self.fited:
            batch = self._init_population(int(self.reject_rate * len(elite_population)))
            best_id = np.argsort(self.rf.predict_log_proba(batch)[:, 0])
            self.population = batch[best_id]
        else:
            self.population = elite_population
        self.population_fitness = np.full(len(self.population), np.inf)