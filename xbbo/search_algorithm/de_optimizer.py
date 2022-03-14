import numpy as np

# from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.search_algorithm.base import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trials, Trial
from . import alg_register


@alg_register.register('de')
class DE(AbstractOptimizer):
    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 llambda=10,
                 boundary_fix_type='random',
                 mutation_factor=0.5,
                 crossover_prob=0.5,
                 strategy='rand1_bin',
                 **kwargs):
        AbstractOptimizer.__init__(self,
                                   space,
                                   encoding_cat='bin',
                                   encoding_ord='bin',
                                   seed=seed,
                                   **kwargs)

        # Uniform2Gaussian.__init__(self,)
        self.dimension = self.space.get_dimensions()
        self.bounds = self.space.get_bounds()
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.fix_type = boundary_fix_type
        # self.dimension = len(configs)
        self.strategy = strategy        
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split('_')[0]
            self.crossover_strategy = self.strategy.split('_')[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.llambda = max(llambda, self._set_min_pop_size())
        self.population = [None] * self.llambda
        self.population_fitness = [np.inf] * self.llambda
        # self.candidates = [None] * self.llambda
        self.trials = Trials(dim=self.dimension)
        self.current_best = None
        self.current_best_fitness = np.inf
        self._num_suggestions = 0
        # self.F1 = kwargs.get('F1', 0.8)
        # self.F2 = kwargs.get('F2', 0.8)
        # self.CR = kwargs.get('CR', 0.5)

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

    def fix_boundary(self, individual):
        if self.fix_type == 'random':
            return np.where(
                (individual > self.bounds.lb) & (individual < self.bounds.ub),
                individual,
                self.rng.uniform(self.bounds.lb, self.bounds.ub,
                                 self.dimension))  # FIXME
        elif self.fix_type == 'clip':
            return np.clip(individual, self.bounds.lb, self.bounds.ub)

    def _suggest(self, n_suggestions=1):
        trial_list = []
        for n in range(n_suggestions):
            # suggest_array = []
            idx = self._num_suggestions % self.llambda
            target = self.population[idx]
            if target is not None:
                donor = self.mutation(current=target, best=None)
                if donor is None:
                    candidate = self.rng.uniform(self.bounds.lb,
                                                    self.bounds.ub,
                                                    self.dimension)
                else:
                    candidate = self.crossover(target, donor)
            else:
                candidate = self.rng.uniform(self.bounds.lb,
                                                    self.bounds.ub,
                                                    self.dimension)


            self._num_suggestions += 1
            candidate = self.fix_boundary(candidate)
            # array = self.feature_to_array(new_individual)
            config = DenseConfiguration.from_array(self.space, candidate)
            trial_list.append(
                Trial(config,
                      config_dict=config.get_dictionary(),
                      array=candidate,
                      origin='DE',
                      loc=idx))

        return trial_list

    def _observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicate=True)
            # idx = self.candidates.index(tuple(trial.array))
            idx = trial.loc
            # selection
            if trial.observe_value <= self.population_fitness[idx]:
                self.population[idx] = trial.array
                self.population_fitness[idx] = trial.observe_value
            # self._num_suggestions += 1
            # self.candidates[idx] = None
                if trial.observe_value < self.current_best_fitness:
                    self.current_best = self.population[idx]
                    self.current_best_fitness = trial.observe_value
                
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


opt_class = DE
