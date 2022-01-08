from typing import Optional, List, Tuple, cast

import numpy as np

from xbbo.configspace.feature_space import Uniform2Gaussian
from xbbo.core import AbstractOptimizer
from xbbo.configspace.space import DenseConfiguration, DenseConfigurationSpace
from xbbo.core.trials import Trial, Trials


class RegularizedEA(AbstractOptimizer):
    '''
    Regularized Evolution for Image Classifier Architecture Search
    '''

    def __init__(self,
                 space: DenseConfigurationSpace,
                 seed: int = 42,
                 llambda=10,
                 **kwargs):
        AbstractOptimizer.__init__(self, space, seed, **kwargs)
        # FeatureSpace_gaussian.__init__(self, self.space.dtypes_idx_map)
        # Uniform2Gaussian.__init__(self)
        # configs = self.space.get_hyperparameters()
        self.dense_dimension = self.space.get_dimensions(sparse=False)
        self.sparse_dimension = self.space.get_dimensions(sparse=True)
        self.bounds = self.space.get_bounds()

        self.num_of_tour_particips = kwargs.get('n_part',2)
        self.llambda = llambda # 4 + math.floor(3 * math.log(n_dim)) # (eq. 48)
        self.population = self.create_initial_population()
        self.population_y = None

        self.mu = kwargs.get('mu',20) # 交叉和变异算法的分布指数
        self.mum = kwargs.get('mum',20)
        self.crossrate = kwargs.get('crossrate',0.9)
        # ---



        self.trials = Trials(sparse_dim=self.sparse_dimension,
                             dense_dim=self.dense_dimension)
        self.cur = 0
        self.gen = 0
        self.listy = []

    def suggest(self, n_suggestions=1):
        assert self.llambda % n_suggestions == 0
        trial_list = []
        for n in range(n_suggestions):
            new_individual = self.population[self.cur]
            new_individual = np.clip(new_individual, self.bounds.lb, self.bounds.ub)
            # dense_array = self.feature_to_array(new_individual)
            config = DenseConfiguration.from_dense_array(self.space,new_individual)
            self.cur += 1
            trial_list.append(
            Trial(config,
                    config_dict=config.get_dictionary(),
                    dense_array=new_individual,
                    origin='REA', loc=self.cur))
        

        # self._num_suggestions += n_suggestions
        return trial_list

    def observe(self, trial_list):
        for trial in trial_list:
            self.trials.add_a_trial(trial, permit_duplicagte=True)
            self.listy.append(trial.observe_value)
        if self.cur == len(self.population):
            if self.population_y is None:
                self.population_y = np.asarray(self.listy)
            else:
                self.population_y = np.concatenate([self.population_y, self.listy], axis=0)

            # remove dead
            self.population_y = np.asarray(self.population_y[-self.llambda:])
            self.population = np.asarray(self.population[-self.llambda:])

            s_id = np.argsort(self.population_y)
            rank = np.argsort(s_id)
            parents_id = list(self.selection_tournament(rank, self.llambda / 2))

            self.children = self.create_children(parents_id)

            # ---
            self.population = self.population[parents_id]
            self.population_y = self.population_y[parents_id]
            self.population = np.append(self.population, self.children, axis=0)
            self.cur = len(self.population_y)
            assert self.cur < len(self.population)

            self.listy = []
            self.gen += 1

    def selection_tournament(self, rank, num):
        # parents = []
        parents_id = set()
        while len(parents_id) < num:
            parent_id = self.__tournament(rank)
            # if parent_id in parents_id:
            #     continue
            parents_id.add(parent_id)
            # parents.append(self.population[parent_id])
        return parents_id

    # def fast_nondominated_sort(self, points):
    #     individual_num = len(points)
    #
    #     return ranks


    def calculate_crowding_distance(self, points, m, M):
        cd = np.zeros(len(points))
        if len(cd) <= points.shape[-1]:
            return cd
        for n in range(points.shape[-1]):
            s_id = np.argsort(points[:, n])
            cd[s_id[[0, -1]]] = [np.inf, np.inf]
            for idx in range(len(s_id[1:-1])):
                cd[s_id[idx]] += (points[s_id[idx+1], n] - points[s_id[idx-1], n]) / (M[n] - m[n])
        return cd



    def create_initial_population(self):
        # return self.rng.normal(0, 1, size=(self.llambda, self.dense_dimension))
        return self.rng.uniform(0, 1, size=(self.llambda, self.dense_dimension))


    def create_children(self, parents_id):
        children = []
        num = len(parents_id)
        list(range(num))
        for i in range(num):
            if self.rng.rand() > self.crossrate:
                child = self.__mutate(self.population[self.rng.choice(parents_id)])
                children.append(child)
            else:
                parent1_id, parent2_id = self.rng.choice(parents_id, 2, replace=False)
                parent1 = np.copy(self.population[parent1_id])
                parent2 = np.copy(self.population[parent2_id])
                child1, child2 = self.__crossover(parent1, parent2)

                children.append(child1)
                children.append(child2)

        return children

    def __crossover(self, individual1, individual2):
        u = self.rng.rand(len(individual1))
        bq = np.where(u<0.5, 2*u**(1/(self.mu+1)), (1 / (2 * (1 - u))) ** (1 / (self.mu + 1)))
        child1 = 0.5 * (((1 + bq) * individual1) + (1 - bq) * individual2)

        child2 = 0.5 * (((1 - bq) * individual1) + (1 + bq) * individual2)

        # child1 = individual1.copy()
        # child2 = individual2.copy()
        # i = self.rng.randint(child1.shape[1])
        # t = child1[i]
        # child1[i] = child2[i]
        # child2[i] = t
        return np.clip(child1, 0, 1), np.clip(child2, 0, 1)

    def __mutate(self, parent):
        child = parent.copy()
        r = self.rng.rand(len(child))
        delta = np.where(r<0.5, (2 * r) ** (1 / (self.mum + 1)) - 1, 1 - (2*(1 - r))**(1/(self.mum+1)))
        return np.clip(child + delta, 0, 1)

    def __tournament(self, rank):
        participants = self.rng.choice(len(self.population), self.num_of_tour_particips, replace=False)


        return participants[np.argsort(rank[participants])[0]]


opt_class = RegularizedEA
